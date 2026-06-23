# Turn Change Appraisal v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace tissue-derived turn novelty with logprob-calibrated `turn_change_appraisal` on every persisted chat turn, feed Spark telemetry from appraisal (never tissue fallback), and emit substrate `organ_signal` molecules on high-confidence novel turns.

**Architecture:** Extend the existing `memory.turn.persisted` → `classify_turn` → `chat.history.spark_meta.patch` rail in `orion-memory-consolidation`. Parsing lives in new `orion/memory/turn_change_classify.py`; the unified four-line LLM prompt reuses `binary_score_from_top_logprobs` from `consolidation_classify.py`. Baseline comparison uses prior window turn (fallback: session window). Substrate feed publishes passthrough `OrionSignalV1` (`signal.memory_consolidation.turn_change`) and extends `signal_bridge` to project it into `organ_signal` molecules. Tissue φ stamping is removed from LLM gateway and spark-introspector hot paths.

**Tech Stack:** Python 3.12, pydantic v2, Orion bus (`OrionBusAsync`, `BaseEnvelope`), asyncpg `WindowStore`, LLM gateway quick lane with logprobs, `OrionSignalV1`, `orion/substrate/signal_bridge.py`.

**Design spec:** `docs/superpowers/specs/2026-06-22-turn-change-appraisal-v1-design.md`

**Parent plan:** `docs/superpowers/plans/2026-06-16-memory-consolidation-pipeline.md`

**Worktree:** Implement in an isolated worktree (`using-superpowers:using-git-worktrees`) before touching main.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `orion/memory/turn_change_classify.py` | Create | Enum softmax, confidence/margin, prompt builders, appraisal dict assembly |
| `orion/memory/turn_change_signal.py` | Create | `OrionSignalV1` builder + shift_kind → dimension gradients |
| `orion/memory/consolidation_classify.py` | Modify | Four-line unified prompt; delegate NOVEL/SHIFT parsing imports |
| `services/orion-memory-consolidation/app/boundary.py` | Modify | Extend `scores_from_llm_result` for four lines |
| `services/orion-memory-consolidation/app/classify.py` | Modify | Baseline selection, fallback second call, patch fields, RPC options |
| `services/orion-memory-consolidation/app/worker.py` | Modify | Prior-turn fetch before classify; substrate emit post-patch |
| `services/orion-memory-consolidation/app/settings.py` | Modify | Three `TURN_CHANGE_*` env fields |
| `services/orion-memory-consolidation/.env_example` | Modify | Document new env keys |
| `orion/signals/registry.py` | Modify | Register `memory_consolidation` organ + `turn_change` kind |
| `orion/substrate/signal_bridge.py` | Modify | Support `(memory_consolidation, turn_change)` in bridge |
| `services/orion-spark-introspector/app/worker.py` | Modify | Telemetry `novelty` from appraisal; remove tissue propagate |
| `services/orion-llm-gateway/app/llm_backend.py` | Modify | Stop tissue φ stamping for chat turns |
| `orion/schemas/telemetry/turn_effect.py` | Modify | `turn_effect_from_appraisal` helper |
| `orion/cognition/prompts/introspect_spark.j2` | Modify | Appraisal table instead of φ delta table |
| `tests/test_turn_change_classify.py` | Create | Logprob fixtures, margin, prompts |
| `tests/test_substrate_signal_bridge.py` | Modify | `turn_change` bridge case |
| `tests/test_turn_effect.py` | Modify | Appraisal-based turn effect |
| `services/orion-memory-consolidation/tests/test_classify_turn_change.py` | Create | Integration patch + substrate gate |
| `tests/test_consolidation_classify.py` | Modify | Four-line prompt assertions |
| `scripts/sync_local_env_from_example.py` | Run after env | Local `.env` sync |

---

### Task 1: Core turn-change scoring (`turn_change_classify.py`)

**Files:**
- Create: `orion/memory/turn_change_classify.py`
- Test: `tests/test_turn_change_classify.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_turn_change_classify.py
import math

import pytest

from orion.memory.turn_change_classify import (
    SHIFT_KINDS,
    appraisal_confidence,
    binary_margin,
    build_change_only_prompt,
    build_turn_change_prompt,
    enum_scores_from_top_logprobs,
    novel_margin_below_threshold,
    parse_novel_shift_lines,
)


def test_enum_scores_from_top_logprobs_softmax():
    tops = [
        {"token": "TOPIC", "logprob": -0.2},
        {"token": "NONE", "logprob": -2.0},
        {"token": "STANCE", "logprob": -1.5},
        {"token": "REPAIR", "logprob": -3.0},
    ]
    scores = enum_scores_from_top_logprobs(tops, SHIFT_KINDS)
    assert scores is not None
    assert set(scores) == set(SHIFT_KINDS)
    assert scores["TOPIC"] == pytest.approx(
        math.exp(-0.2) / sum(math.exp(-x) for x in (-0.2, -2.0, -1.5, -3.0)),
        rel=1e-3,
    )
    assert scores["TOPIC"] > scores["NONE"]


def test_appraisal_confidence_min_margin():
    assert appraisal_confidence(0.82, 0.78) == pytest.approx(0.64, rel=1e-3)
    assert appraisal_confidence(0.52) == pytest.approx(0.04, rel=1e-3)


def test_novel_margin_below_threshold():
    assert novel_margin_below_threshold(0.52, margin=0.15) is True
    assert novel_margin_below_threshold(0.82, margin=0.15) is False


def test_parse_novel_shift_lines():
    text = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: NO\nBOUNDARY: NO\n"
    novel, shift = parse_novel_shift_lines(text)
    assert novel == "YES"
    assert shift == "TOPIC"


def test_build_turn_change_prompt_includes_baseline():
    p = build_turn_change_prompt(
        prompt="new topic",
        response="sure",
        baseline_mode="prior_turn",
        baseline_text="User: old\nOrion: prior\n",
        phase="same_breath",
    )
    assert "NOVEL:" in p
    assert "SHIFT:" in p
    assert "prior_turn" in p or "User: old" in p


def test_build_change_only_prompt_two_lines():
    p = build_change_only_prompt(
        prompt="p",
        response="r",
        baseline_text="User: a\nOrion: b\n",
    )
    assert "NOVEL:" in p
    assert "SHIFT:" in p
    assert "MEMORY:" not in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_turn_change_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: orion.memory.turn_change_classify`

- [ ] **Step 3: Write minimal implementation**

```python
# orion/memory/turn_change_classify.py
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

_NOVEL_LINE = re.compile(r"^NOVEL:\s*(YES|NO)\s*$", re.I | re.M)
_SHIFT_LINE = re.compile(r"^SHIFT:\s*(NONE|TOPIC|STANCE|REPAIR)\s*$", re.I | re.M)

SHIFT_KINDS: tuple[str, ...] = ("NONE", "TOPIC", "STANCE", "REPAIR")


def enum_scores_from_top_logprobs(
    top_logprobs: list[dict],
    tokens: tuple[str, ...],
) -> dict[str, float] | None:
    token_set = {t.upper() for t in tokens}
    lps: dict[str, float] = {}
    for entry in top_logprobs or []:
        tok = str(entry.get("token") or "").strip().upper()
        lp = entry.get("logprob")
        if tok in token_set and isinstance(lp, (int, float)):
            lps[tok] = float(lp)
    if not lps:
        return None
    exps = {k: math.exp(v) for k, v in lps.items()}
    total = sum(exps.values())
    if total <= 0:
        return None
    return {k: exps[k] / total for k in tokens if k in exps}


def binary_margin(score: float | None) -> float | None:
    if score is None:
        return None
    return abs(float(score) - 0.5)


def appraisal_confidence(*scores: float | None) -> float | None:
    margins = [2.0 * abs(float(s) - 0.5) for s in scores if s is not None]
    return min(margins) if margins else None


def novel_margin_below_threshold(novelty_score: float | None, *, margin: float) -> bool:
    m = binary_margin(novelty_score)
    return m is not None and m < float(margin)


def parse_novel_shift_lines(text: str) -> tuple[str | None, str | None]:
    novel = _NOVEL_LINE.search(text or "")
    shift = _SHIFT_LINE.search(text or "")
    return (
        novel.group(1).upper() if novel else None,
        shift.group(1).upper() if shift else None,
    )


def _clip_pair(prompt: str, response: str, *, limit: int = 300) -> tuple[str, str]:
    def _c(s: str) -> str:
        s = (s or "").strip()
        return s if len(s) <= limit else s[: limit - 3] + "..."

    return _c(prompt), _c(response)


def build_turn_change_prompt(
    *,
    prompt: str,
    response: str,
    baseline_mode: str,
    baseline_text: str,
    phase: str = "unknown",
) -> str:
    p, r = _clip_pair(prompt, response)
    return (
        "Classify this turn vs the baseline. Output exactly four lines:\n"
        "NOVEL: YES or NO\n"
        "SHIFT: NONE or TOPIC or STANCE or REPAIR\n"
        "MEMORY: YES or NO\n"
        "BOUNDARY: YES or NO\n\n"
        f"baseline_mode={baseline_mode}\n"
        f"phase={phase}\n"
        f"--- BASELINE ---\n{baseline_text.strip()}\n"
        f"--- CURRENT ---\nUser: {p!r}\nOrion: {r!r}\n"
    )


def build_change_only_prompt(
    *,
    prompt: str,
    response: str,
    baseline_text: str,
    phase: str = "unknown",
) -> str:
    p, r = _clip_pair(prompt, response)
    return (
        "Re-appraise change vs session window baseline. Output exactly two lines:\n"
        "NOVEL: YES or NO\n"
        "SHIFT: NONE or TOPIC or STANCE or REPAIR\n\n"
        f"phase={phase}\n"
        f"--- SESSION WINDOW BASELINE ---\n{baseline_text.strip()}\n"
        f"--- CURRENT ---\nUser: {p!r}\nOrion: {r!r}\n"
    )


def build_turn_change_appraisal(
    *,
    baseline_mode: str,
    prior_correlation_id: str | None,
    novelty_score: float | None,
    shift_kind: str | None,
    shift_scores: dict[str, float] | None,
    confidence: float | None,
    status: str,
) -> dict[str, Any]:
    return {
        "baseline_mode": baseline_mode,
        "prior_correlation_id": prior_correlation_id,
        "novelty_score": novelty_score,
        "shift_kind": shift_kind,
        "shift_scores": shift_scores or {},
        "confidence": confidence,
        "turn_change_status": status,
        "turn_change_ts": datetime.now(timezone.utc).isoformat(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_turn_change_classify.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/memory/turn_change_classify.py tests/test_turn_change_classify.py
git commit -m "feat: add turn change classify scoring primitives"
```

---

### Task 2: Extend consolidation classify prompt (remove tissue φ)

**Files:**
- Modify: `orion/memory/consolidation_classify.py`
- Test: `tests/test_consolidation_classify.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_consolidation_classify.py`:

```python
def test_build_classify_prompt_four_lines_no_phi():
    p = build_classify_prompt(
        prompt="hi",
        response="hello",
        spark_meta={"conversation_phase": {"phase_change": "same_breath"}},
        baseline_mode="prior_turn",
        baseline_text="User: earlier\nOrion: reply\n",
    )
    assert "NOVEL:" in p
    assert "SHIFT:" in p
    assert "MEMORY:" in p
    assert "BOUNDARY:" in p
    assert "phi_after" not in p
    assert "novelty=" not in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_consolidation_classify.py::test_build_classify_prompt_four_lines_no_phi -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'baseline_mode'`)

- [ ] **Step 3: Update `build_classify_prompt`**

Replace `build_classify_prompt` in `orion/memory/consolidation_classify.py`:

```python
from orion.memory.turn_change_classify import build_turn_change_prompt


def build_classify_prompt(
    *,
    prompt: str,
    response: str,
    spark_meta: dict[str, Any],
    baseline_mode: str = "none",
    baseline_text: str = "",
) -> str:
    phase = (
        (spark_meta.get("conversation_phase") or {}).get("phase_change")
        or spark_meta.get("temporal_phase")
        or "unknown"
    )
    return build_turn_change_prompt(
        prompt=prompt,
        response=response,
        baseline_mode=baseline_mode,
        baseline_text=baseline_text,
        phase=str(phase),
    )
```

Update existing `test_build_classify_prompt_includes_phase` to pass `baseline_text=""` and assert `NOVEL:` present.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_consolidation_classify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/memory/consolidation_classify.py tests/test_consolidation_classify.py
git commit -m "feat: unify classify prompt to four-line turn change format"
```

---

### Task 3: Four-line logprob parsing in `boundary.py`

**Files:**
- Modify: `services/orion-memory-consolidation/app/boundary.py`
- Test: `services/orion-memory-consolidation/tests/test_boundary.py`

- [ ] **Step 1: Write the failing test**

Add to `services/orion-memory-consolidation/tests/test_boundary.py`:

```python
def test_scores_from_llm_result_four_lines():
    content = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: YES\nBOUNDARY: NO\n"
    raw = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "NOVEL:", "logprob": -0.1},
                        {
                            "token": "YES",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "YES", "logprob": -0.2},
                                {"token": "NO", "logprob": -2.0},
                            ],
                        },
                        {"token": "SHIFT:", "logprob": -0.1},
                        {
                            "token": "TOPIC",
                            "logprob": -0.3,
                            "top_logprobs": [
                                {"token": "TOPIC", "logprob": -0.3},
                                {"token": "NONE", "logprob": -2.0},
                                {"token": "STANCE", "logprob": -2.5},
                                {"token": "REPAIR", "logprob": -3.0},
                            ],
                        },
                        {"token": "MEMORY:", "logprob": -0.1},
                        {
                            "token": "YES",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "YES", "logprob": -0.2},
                                {"token": "NO", "logprob": -2.0},
                            ],
                        },
                        {"token": "BOUNDARY:", "logprob": -0.1},
                        {
                            "token": "NO",
                            "logprob": -0.3,
                            "top_logprobs": [
                                {"token": "NO", "logprob": -0.3},
                                {"token": "YES", "logprob": -1.5},
                            ],
                        },
                    ]
                }
            }
        ]
    }
    result = scores_from_llm_result(content, raw)
    assert result["novelty_score"] == pytest.approx(0.83, abs=0.05)
    assert result["shift_kind"] == "TOPIC"
    assert result["shift_scores"]["TOPIC"] > result["shift_scores"]["NONE"]
    assert result["memory_significance_score"] is not None
    assert result["conversation_boundary_score"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./scripts/test_service.sh orion-memory-consolidation services/orion-memory-consolidation/tests/test_boundary.py::test_scores_from_llm_result_four_lines -v`
Expected: FAIL (`TypeError: cannot unpack` or missing keys)

- [ ] **Step 3: Replace `scores_from_llm_result`**

```python
# services/orion-memory-consolidation/app/boundary.py
from orion.memory.turn_change_classify import (
    SHIFT_KINDS,
    appraisal_confidence,
    enum_scores_from_top_logprobs,
    parse_novel_shift_lines,
)


def scores_from_llm_result(content: str, raw: dict[str, Any]) -> dict[str, Any]:
    logprobs = (((raw.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or [])
    mem_score = bnd_score = novelty_score = None
    shift_scores: dict[str, float] | None = None
    shift_kind: str | None = None
    line = "novel"

    for entry in logprobs:
        tok = str(entry.get("token") or "").strip().upper()
        if tok == "NOVEL:":
            line = "novel"
            continue
        if tok == "SHIFT:":
            line = "shift"
            continue
        if tok == "MEMORY:":
            line = "memory"
            continue
        if tok == "BOUNDARY:":
            line = "boundary"
            continue

        tops = entry.get("top_logprobs") or [{"token": tok, "logprob": entry.get("logprob")}]

        if line == "novel" and tok in ("YES", "NO"):
            novelty_score = binary_score_from_top_logprobs(tops)
        elif line == "shift" and tok in SHIFT_KINDS:
            shift_scores = enum_scores_from_top_logprobs(tops, SHIFT_KINDS)
            if shift_scores:
                shift_kind = max(shift_scores, key=shift_scores.get)
        elif line == "memory" and tok in ("YES", "NO"):
            mem_score = binary_score_from_top_logprobs(tops)
        elif line == "boundary" and tok in ("YES", "NO"):
            bnd_score = binary_score_from_top_logprobs(tops)

    novel_yes, shift_txt = parse_novel_shift_lines(content)
    mem_yes, bnd_yes = parse_classify_lines(content)

    if novelty_score is None and novel_yes:
        novelty_score = 0.85 if novel_yes == "YES" else 0.15
    if shift_kind is None and shift_txt:
        shift_kind = shift_txt
    if mem_score is None and mem_yes:
        mem_score = 0.85 if mem_yes == "YES" else 0.15
    if bnd_score is None and bnd_yes:
        bnd_score = 0.85 if bnd_yes == "YES" else 0.15

    confidence = appraisal_confidence(novelty_score, shift_scores.get(shift_kind) if shift_scores and shift_kind else None)

    return {
        "novelty_score": novelty_score,
        "shift_kind": shift_kind,
        "shift_scores": shift_scores or {},
        "confidence": confidence,
        "memory_significance_score": mem_score,
        "conversation_boundary_score": bnd_score,
    }
```

Update `test_scores_from_llm_result_uses_logprobs` to unpack dict keys instead of tuple.

- [ ] **Step 4: Run boundary tests**

Run: `./scripts/test_service.sh orion-memory-consolidation services/orion-memory-consolidation/tests/test_boundary.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-memory-consolidation/app/boundary.py services/orion-memory-consolidation/tests/test_boundary.py
git commit -m "feat: parse four-line classify logprobs for turn change"
```

---

### Task 4: Settings + env contract

**Files:**
- Modify: `services/orion-memory-consolidation/app/settings.py`
- Modify: `services/orion-memory-consolidation/.env_example`

- [ ] **Step 1: Add settings fields**

```python
# services/orion-memory-consolidation/app/settings.py (after MEMORY_CLASSIFY_TIMEOUT_SEC)
TURN_CHANGE_CONFIDENCE_MARGIN: float = Field(default=0.15, alias="TURN_CHANGE_CONFIDENCE_MARGIN")
TURN_CHANGE_SUBSTRATE_THRESHOLD: float = Field(default=0.65, alias="TURN_CHANGE_SUBSTRATE_THRESHOLD")
TURN_CHANGE_WINDOW_TURNS: int = Field(default=3, alias="TURN_CHANGE_WINDOW_TURNS")
CHANNEL_SIGNALS_PREFIX: str = Field(default="orion:signals", alias="CHANNEL_SIGNALS_PREFIX")
```

- [ ] **Step 2: Update `.env_example`**

```bash
TURN_CHANGE_CONFIDENCE_MARGIN=0.15
TURN_CHANGE_SUBSTRATE_THRESHOLD=0.65
TURN_CHANGE_WINDOW_TURNS=3
CHANNEL_SIGNALS_PREFIX=orion:signals
```

- [ ] **Step 3: Sync local env**

Run: `cd /mnt/scripts/Orion-Sapienform && python scripts/sync_local_env_from_example.py`
Expected: exit 0; `services/orion-memory-consolidation/.env` contains the three `TURN_CHANGE_*` keys.

- [ ] **Step 4: Commit**

```bash
git add services/orion-memory-consolidation/app/settings.py services/orion-memory-consolidation/.env_example
git commit -m "chore: add turn change appraisal env settings"
```

---

### Task 5: Baseline selection + `classify_turn` orchestration

**Files:**
- Modify: `services/orion-memory-consolidation/app/classify.py`
- Create: `services/orion-memory-consolidation/tests/test_classify_turn_change.py`

- [ ] **Step 1: Write failing integration tests**

```python
# services/orion-memory-consolidation/tests/test_classify_turn_change.py
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    path = SERVICE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


classify_mod = _load("app/classify.py", "memory_classify")
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1


def _llm_raw(content: str, *, novel_lp=-0.2, shift_token="NONE") -> dict:
    return {
        "content": content,
        "raw": {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "NOVEL:", "logprob": -0.1},
                            {
                                "token": "YES" if "YES" in content else "NO",
                                "logprob": novel_lp,
                                "top_logprobs": [
                                    {"token": "YES", "logprob": -0.2},
                                    {"token": "NO", "logprob": -2.0},
                                ],
                            },
                            {"token": "SHIFT:", "logprob": -0.1},
                            {
                                "token": shift_token,
                                "logprob": -0.3,
                                "top_logprobs": [
                                    {"token": shift_token, "logprob": -0.3},
                                    {"token": "NONE", "logprob": -2.0},
                                    {"token": "TOPIC", "logprob": -2.5},
                                    {"token": "STANCE", "logprob": -3.0},
                                ],
                            },
                            {"token": "MEMORY:", "logprob": -0.1},
                            {"token": "NO", "logprob": -0.3, "top_logprobs": [{"token": "NO", "logprob": -0.3}, {"token": "YES", "logprob": -2.0}]},
                            {"token": "BOUNDARY:", "logprob": -0.1},
                            {"token": "NO", "logprob": -0.3, "top_logprobs": [{"token": "NO", "logprob": -0.3}, {"token": "YES", "logprob": -2.0}]},
                        ]
                    }
                }
            ]
        },
    }


@pytest.mark.asyncio
async def test_classify_turn_first_turn_baseline_none():
    bus = AsyncMock()
    bus.codec.decode.return_value.ok = True
    turn = MemoryTurnPersistedV1(correlation_id=str(uuid4()), prompt="hi", response="hello", spark_meta={})
    settings = classify_mod.settings
    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=[], settings=settings)
    assert "turn_change_appraisal" in patch
    assert patch["turn_change_appraisal"]["baseline_mode"] == "none"
    assert patch["turn_change_appraisal"]["novelty_score"] is None
    bus.rpc_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_classify_turn_routine_followup_low_novelty(monkeypatch):
    bus = AsyncMock()
    bus.codec.decode.return_value.ok = True
    content = "NOVEL: NO\nSHIFT: NONE\nMEMORY: NO\nBOUNDARY: NO\n"
    bus.rpc_request.return_value = {"data": b"x"}

    async def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, novel_lp=-2.0, shift_token="NONE")})()
        return _R()

    bus.codec.decode.side_effect = _decode_side_effect
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(correlation_id=str(uuid4()), prompt="more cats", response="still cute", spark_meta={})
    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=classify_mod.settings)
    appr = patch["turn_change_appraisal"]
    assert appr["turn_change_status"] == "ok"
    assert appr["baseline_mode"] == "prior_turn"
    assert appr["novelty_score"] < 0.5


@pytest.mark.asyncio
async def test_classify_turn_topic_pivot_high_novelty(monkeypatch):
    bus = AsyncMock()
    content = "NOVEL: YES\nSHIFT: TOPIC\nMEMORY: YES\nBOUNDARY: NO\n"
    bus.rpc_request.return_value = {"data": b"x"}

    async def _decode_side_effect(_):
        class _R:
            ok = True
            envelope = type("E", (), {"payload": _llm_raw(content, shift_token="TOPIC")})()
        return _R()

    bus.codec.decode.side_effect = _decode_side_effect
    prior = [{"correlation_id": "prev", "prompt": "cats", "response": "cute"}]
    turn = MemoryTurnPersistedV1(correlation_id=str(uuid4()), prompt="let's talk kubernetes", response="ok", spark_meta={})
    patch = await classify_mod.classify_turn(bus, turn=turn, prior_turns=prior, settings=classify_mod.settings)
    appr = patch["turn_change_appraisal"]
    assert appr["shift_kind"] == "TOPIC"
    assert appr["novelty_score"] >= 0.65
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test_service.sh orion-memory-consolidation services/orion-memory-consolidation/tests/test_classify_turn_change.py -v`
Expected: FAIL (`TypeError: classify_turn() got an unexpected keyword argument 'prior_turns'`)

- [ ] **Step 3: Implement `classify.py`**

Key helpers to add at top of `services/orion-memory-consolidation/app/classify.py`:

```python
from orion.memory.consolidation_classify import build_classify_prompt
from orion.memory.turn_change_classify import (
    build_change_only_prompt,
    build_turn_change_appraisal,
    novel_margin_below_threshold,
)
from app.boundary import scores_from_llm_result
from app.worker import build_window_transcript  # or duplicate small clip helper to avoid cycle — prefer inline clip in classify.py


def _prior_turn_baseline(prior_turns: list[dict]) -> tuple[str, str, str | None]:
    if not prior_turns:
        return "none", "", None
    last = prior_turns[-1]
    text = f"User: {last.get('prompt', '')}\nOrion: {last.get('response', '')}\n"
    return "prior_turn", text, str(last.get("correlation_id") or "")


def _session_window_baseline(prior_turns: list[dict], *, n: int) -> tuple[str, str]:
    selected = prior_turns[-n:] if len(prior_turns) > n else prior_turns
    if not selected:
        return "none", ""
    return "session_window", build_window_transcript(selected)


async def _llm_classify(bus, *, prompt: str, settings, timeout_key="MEMORY_CLASSIFY_TIMEOUT_SEC") -> dict:
    # existing rpc body; options max_tokens=16, logprobs_top_k=4
    ...
    scores = scores_from_llm_result(content, raw)
    return scores


async def reappraise_with_session_window(bus, *, turn, prior_turns, settings) -> dict:
    mode, text = _session_window_baseline(prior_turns, n=settings.TURN_CHANGE_WINDOW_TURNS)
    if mode == "none":
        return {}
    prompt = build_change_only_prompt(
        prompt=turn.prompt,
        response=turn.response,
        baseline_text=text,
        phase=(turn.spark_meta.get("conversation_phase") or {}).get("phase_change") or "unknown",
    )
    return await _llm_classify(bus, prompt=prompt, settings=settings)


async def classify_turn(bus, *, turn, prior_turns: list[dict], settings) -> dict:
    baseline_mode, baseline_text, prior_corr = _prior_turn_baseline(prior_turns)
    if baseline_mode == "none":
        appraisal = build_turn_change_appraisal(
            baseline_mode="none",
            prior_correlation_id=None,
            novelty_score=None,
            shift_kind=None,
            shift_scores=None,
            confidence=None,
            status="ok",
        )
        return {
            "turn_change_appraisal": appraisal,
            "memory_classify_status": "degraded",
            "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
        }

    phase = (turn.spark_meta.get("conversation_phase") or {}).get("phase_change") or "unknown"
    prompt = build_classify_prompt(
        prompt=turn.prompt,
        response=turn.response,
        spark_meta=turn.spark_meta,
        baseline_mode=baseline_mode,
        baseline_text=baseline_text,
    )

    try:
        scores = await _llm_classify(bus, prompt=prompt, settings=settings)
    except Exception:
        return _degraded_patch()

    novelty = scores.get("novelty_score")
    if novel_margin_below_threshold(novelty, margin=settings.TURN_CHANGE_CONFIDENCE_MARGIN):
        retry = await reappraise_with_session_window(bus, turn=turn, prior_turns=prior_turns, settings=settings)
        if retry:
            scores["novelty_score"] = retry.get("novelty_score", novelty)
            scores["shift_kind"] = retry.get("shift_kind", scores.get("shift_kind"))
            scores["shift_scores"] = retry.get("shift_scores", scores.get("shift_scores"))
            baseline_mode = "session_window"
            _, baseline_text = _session_window_baseline(prior_turns, n=settings.TURN_CHANGE_WINDOW_TURNS)
            prior_corr = None

    status = "ok" if scores.get("novelty_score") is not None else "degraded"
    appraisal = build_turn_change_appraisal(
        baseline_mode=baseline_mode,
        prior_correlation_id=prior_corr,
        novelty_score=scores.get("novelty_score"),
        shift_kind=scores.get("shift_kind"),
        shift_scores=scores.get("shift_scores"),
        confidence=scores.get("confidence"),
        status=status,
    )
    return {
        "turn_change_appraisal": appraisal,
        "memory_significance_score": scores.get("memory_significance_score"),
        "conversation_boundary_score": scores.get("conversation_boundary_score"),
        "memory_classify_status": "ok" if status == "ok" else "degraded",
        "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
    }
```

**Cycle note:** Do not import `build_window_transcript` from `worker.py` if it creates an import cycle. Copy the 10-line clip/transcript helper into `classify.py` or a new `app/transcript.py`.

Update `_llm_classify` RPC options:

```python
options={
    "return_logprobs": True,
    "logprobs_top_k": 4,
    "logprob_summary_only": False,
    "max_tokens": 16,
    "llm_route": "quick",
},
```

- [ ] **Step 4: Run classify tests**

Run: `./scripts/test_service.sh orion-memory-consolidation services/orion-memory-consolidation/tests/test_classify_turn_change.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-memory-consolidation/app/classify.py services/orion-memory-consolidation/tests/test_classify_turn_change.py
git commit -m "feat: classify turn change with baseline fallback"
```

---

### Task 6: Substrate signal builder + bridge extension

**Files:**
- Create: `orion/memory/turn_change_signal.py`
- Modify: `orion/substrate/signal_bridge.py`
- Modify: `orion/signals/registry.py`
- Modify: `tests/test_substrate_signal_bridge.py`

- [ ] **Step 1: Write failing bridge test**

Add to `tests/test_substrate_signal_bridge.py`:

```python
def test_turn_change_signal_converts_to_organ_signal_molecule():
    from datetime import datetime, timezone
    from orion.memory.turn_change_signal import build_turn_change_signal
    from orion.substrate.signal_bridge import signal_to_molecule

    signal = build_turn_change_signal(
        correlation_id="corr-abc",
        shift_kind="TOPIC",
        novelty_score=0.82,
        confidence=0.91,
    )
    molecule = signal_to_molecule(signal)
    assert molecule.molecule_kind == "organ_signal"
    assert molecule.gradients["novelty"] > 0.5
    assert molecule.gradients["salience"] > 0.5
    assert molecule.provenance["source_event_id"] == "corr-abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_substrate_signal_bridge.py::test_turn_change_signal_converts_to_organ_signal_molecule -v`
Expected: FAIL

- [ ] **Step 3: Implement signal builder**

```python
# orion/memory/turn_change_signal.py
from __future__ import annotations

from datetime import datetime, timezone

from orion.signals.models import OrganClass, OrionSignalV1
from orion.signals.signal_ids import make_signal_id


def dimensions_for_shift(*, shift_kind: str, novelty_score: float) -> dict[str, float]:
    ns = max(0.0, min(1.0, float(novelty_score)))
    if shift_kind == "TOPIC":
        return {"novelty": ns, "salience": ns}
    if shift_kind == "STANCE":
        return {"contradiction": ns, "salience": ns}
    if shift_kind == "REPAIR":
        return {"contradiction": ns}
    # NONE + high NOVEL: deadband salience only
    return {"salience": min(0.15, ns * 0.2)}


def build_turn_change_signal(
    *,
    correlation_id: str,
    shift_kind: str,
    novelty_score: float,
    confidence: float,
) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    dims = dimensions_for_shift(shift_kind=shift_kind, novelty_score=novelty_score)
    dims["confidence"] = max(0.0, min(1.0, float(confidence)))
    return OrionSignalV1(
        signal_id=make_signal_id("memory_consolidation", correlation_id),
        organ_id="memory_consolidation",
        organ_class=OrganClass.endogenous,
        signal_kind="turn_change",
        dimensions=dims,
        causal_parents=[],
        source_event_id=correlation_id,
        observed_at=now,
        emitted_at=now,
        summary=f"turn_change shift={shift_kind} novelty={novelty_score:.2f}",
        notes=[],
    )
```

- [ ] **Step 4: Extend `signal_bridge.py`**

```python
SUPPORTED_SIGNAL_KINDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("cortex_exec", "cognition_run"),
        ("cortex_exec", "cognition_step"),
        ("memory_consolidation", "turn_change"),
    }
)
```

No other bridge logic changes needed — `dimensions_to_gradients` already maps `novelty`, `salience`, `contradiction`.

- [ ] **Step 5: Register organ in `orion/signals/registry.py`**

```python
"memory_consolidation": OrionOrganRegistryEntry(
    organ_id="memory_consolidation",
    organ_class=OrganClass.endogenous,
    service="orion-memory-consolidation",
    signal_kinds=["turn_change"],
    canonical_dimensions=["novelty", "salience", "contradiction", "confidence"],
    causal_parent_organs=["hub"],
    bus_channels=["orion:memory:turn:persisted"],
    notes=["Logprob turn change appraisal substrate perturbation."],
),
```

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_substrate_signal_bridge.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add orion/memory/turn_change_signal.py orion/substrate/signal_bridge.py orion/signals/registry.py tests/test_substrate_signal_bridge.py
git commit -m "feat: bridge memory_consolidation turn_change signals to substrate"
```

---

### Task 7: Worker wiring — prior turns, patch, substrate emit

**Files:**
- Modify: `services/orion-memory-consolidation/app/worker.py`
- Modify: `services/orion-memory-consolidation/tests/test_worker_classify_patch.py`

- [ ] **Step 1: Write failing substrate gate test**

Add to `services/orion-memory-consolidation/tests/test_classify_turn_change.py`:

```python
@pytest.mark.asyncio
async def test_worker_emits_substrate_signal_above_threshold(monkeypatch):
    worker = _load("app/worker.py", "memory_consolidation_worker")
    published = []

    bus = AsyncMock()

    async def _publish(channel, env):
        published.append((channel, env))

    bus.publish = _publish

    appraisal = {
        "turn_change_status": "ok",
        "novelty_score": 0.9,
        "shift_kind": "TOPIC",
        "confidence": 0.88,
    }

    async def _fake_classify(bus, *, turn, prior_turns, settings):
        return {
            "turn_change_appraisal": appraisal,
            "memory_significance_score": 0.5,
            "conversation_boundary_score": 0.1,
            "memory_classify_status": "ok",
        }

    monkeypatch.setattr(worker, "classify_turn", _fake_classify)
    # ... invoke handle_memory_turn_persisted ...
    signal_kinds = [env.kind for _, env in published]
    assert "signal.memory_consolidation.turn_change" in signal_kinds
```

- [ ] **Step 2: Update `handle_memory_turn_persisted`**

```python
async def _maybe_publish_turn_change_signal(
    bus: OrionBusAsync,
    *,
    correlation_id: str,
    appraisal: dict[str, Any],
) -> None:
    from orion.memory.turn_change_signal import build_turn_change_signal

    if appraisal.get("turn_change_status") != "ok":
        return
    novelty = appraisal.get("novelty_score")
    if not isinstance(novelty, (int, float)) or float(novelty) < settings.TURN_CHANGE_SUBSTRATE_THRESHOLD:
        return
    shift_kind = str(appraisal.get("shift_kind") or "NONE")
    confidence = float(appraisal.get("confidence") or 0.0)
    signal = build_turn_change_signal(
        correlation_id=correlation_id,
        shift_kind=shift_kind,
        novelty_score=float(novelty),
        confidence=confidence,
    )
    env = BaseEnvelope(
        kind="signal.memory_consolidation.turn_change",
        correlation_id=correlation_id,
        source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
        payload=signal.model_dump(mode="json"),
    )
    channel = f"{settings.CHANNEL_SIGNALS_PREFIX}:memory_consolidation"
    await bus.publish(channel, env)


async def handle_memory_turn_persisted(...):
    turn = MemoryTurnPersistedV1.model_validate(env.payload)
    open_row = await window_store._get_open_window()
    prior_turns = (
        await window_store.get_window_turns(open_row["memory_window_id"])
        if open_row is not None
        else []
    )
    patch_fields = await classify_turn(bus, turn=turn, prior_turns=prior_turns, settings=settings)
    await publish_spark_meta_patch(bus, turn.correlation_id, patch_fields)
    await _maybe_publish_turn_change_signal(
        bus,
        correlation_id=turn.correlation_id,
        appraisal=patch_fields.get("turn_change_appraisal") or {},
    )
    await window_store.append_turn(turn, scores=patch_fields)
    ...
```

- [ ] **Step 3: Fix `test_worker_classify_patch` fake classify signature**

Update `_fake_classify` to accept `prior_turns=[]`.

- [ ] **Step 4: Run worker tests**

Run: `./scripts/test_service.sh orion-memory-consolidation services/orion-memory-consolidation/tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-memory-consolidation/app/worker.py services/orion-memory-consolidation/tests/
git commit -m "feat: emit substrate turn_change signal after appraisal patch"
```

---

### Task 8: Spark introspector — appraisal novelty, remove tissue hot path

**Files:**
- Modify: `services/orion-spark-introspector/app/worker.py`

- [ ] **Step 1: Add appraisal novelty extractor**

```python
def _novelty_from_spark_meta(spark_meta: Dict[str, Any]) -> Optional[float]:
    if not isinstance(spark_meta, dict):
        return None
    appraisal = spark_meta.get("turn_change_appraisal")
    if isinstance(appraisal, dict) and appraisal.get("turn_change_status") == "ok":
        score = appraisal.get("novelty_score")
        if isinstance(score, (int, float)):
            return float(score)
    # degraded: explicit null — never tissue fallback
    if isinstance(appraisal, dict) and appraisal.get("turn_change_status") == "degraded":
        return None
    return None
```

Replace `_extract_phi_novelty_from_meta` call sites for telemetry `novelty` field with `_novelty_from_spark_meta` first; keep phi coherence path only if still needed for non-novelty metrics.

- [ ] **Step 2: Update `_candidate_quality`**

Add `"turn_change_appraisal"` to `rich_keys` tuple.

- [ ] **Step 3: Remove tissue propagate from candidate handler (~lines 831–843)**

Delete the block:

```python
stimulus = MAPPER.surface_to_stimulus(encoding, magnitude=1.0)
novelty = float(TISSUE.calculate_novelty(stimulus, channel_key="chat"))
TISSUE.propagate(...)
```

Replace with static snapshot using `_get_phi_stats()` only (no propagate). Set telemetry novelty from `_novelty_from_spark_meta(c.spark_meta)`.

- [ ] **Step 4: Remove tissue propagate from `handle_trace` non-heartbeat path (~lines 1072–1073)**

Keep heartbeat path tissue-free already; for cognition traces, skip `TISSUE.calculate_novelty` + `TISSUE.propagate`. Continue emitting telemetry using appraisal when present in trace spark_meta (if available) else `novelty=None`.

- [ ] **Step 5: Manual compile check**

Run: `python -m compileall services/orion-spark-introspector/app/worker.py`
Expected: exit 0

- [ ] **Step 6: Commit**

```bash
git add services/orion-spark-introspector/app/worker.py
git commit -m "feat: spark telemetry novelty from turn_change_appraisal, drop tissue propagate"
```

---

### Task 9: LLM gateway — stop tissue φ stamping

**Files:**
- Modify: `services/orion-llm-gateway/app/llm_backend.py`

- [ ] **Step 1: Gut `_spark_ingest_for_body`**

Return non-tissue metadata only:

```python
def _spark_ingest_for_body(body: ChatBody) -> Dict[str, Any]:
    try:
        messages = _serialize_messages(body.messages or [])
        if not messages:
            return {}
        source = getattr(body, "source", "llm-gateway")
        verb = getattr(body, "verb", None) or "unknown"
        raw_user_text = _get_raw_user_text(body)
        latest_user = raw_user_text
        if not latest_user:
            for m in reversed(messages):
                if (m.get("role") or "").lower() == "user":
                    latest_user = m.get("content")
                    break
        if not latest_user:
            latest_user = messages[-1].get("content")
        if not latest_user:
            return {}
        return {
            "latest_user_message": str(latest_user),
            "trace_verb": verb,
            "spark_phase": "pre",
            "spark_used_raw_user_text": bool(raw_user_text),
        }
    except Exception as e:
        logger.warning(f"[LLM-GW Spark] Ingestion failed: {e}")
        return {}
```

- [ ] **Step 2: Gut `_spark_post_ingest_for_reply`**

Keep only `latest_assistant_message` clip — remove `phi_post_before` / `phi_post_after` writes.

- [ ] **Step 3: Update `_maybe_publish_spark_introspect` trigger**

Replace phi delta check with: publish when `latest_user_message` present (existing introspect path); remove dependency on `phi_before`/`phi_after` delta.

- [ ] **Step 4: Compile**

Run: `python -m compileall services/orion-llm-gateway/app/llm_backend.py`
Expected: exit 0

- [ ] **Step 5: Commit**

```bash
git add services/orion-llm-gateway/app/llm_backend.py
git commit -m "refactor: remove tissue phi stamping from llm gateway chat path"
```

---

### Task 10: `turn_effect_from_appraisal` telemetry helper

**Files:**
- Modify: `orion/schemas/telemetry/turn_effect.py`
- Modify: `tests/test_turn_effect.py`

- [ ] **Step 1: Write failing test**

```python
def test_turn_effect_from_appraisal():
    from orion.schemas.telemetry.turn_effect import turn_effect_from_appraisal

    spark_meta = {
        "turn_change_appraisal": {
            "turn_change_status": "ok",
            "novelty_score": 0.82,
            "shift_kind": "TOPIC",
            "confidence": 0.91,
        }
    }
    effect = turn_effect_from_appraisal(spark_meta)
    assert effect is not None
    assert effect["turn"]["novelty"] == pytest.approx(0.82)
    assert effect["evidence"]["turn_change_appraisal"]["shift_kind"] == "TOPIC"


def test_turn_effect_from_appraisal_degraded_returns_none_novelty():
    from orion.schemas.telemetry.turn_effect import turn_effect_from_appraisal

    effect = turn_effect_from_appraisal({"turn_change_appraisal": {"turn_change_status": "degraded"}})
    assert effect is None or effect.get("turn", {}).get("novelty") is None
```

- [ ] **Step 2: Implement**

```python
def turn_effect_from_appraisal(spark_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(spark_meta, dict):
        return None
    appraisal = spark_meta.get("turn_change_appraisal")
    if not isinstance(appraisal, dict):
        return None
    if appraisal.get("turn_change_status") != "ok":
        return None
    novelty = _coerce_float(appraisal.get("novelty_score"))
    if novelty is None:
        return None
    effect: Dict[str, Any] = {
        "turn": {"novelty": novelty},
        "evidence": {"turn_change_appraisal": {
            k: appraisal[k] for k in ("shift_kind", "shift_scores", "confidence", "baseline_mode")
            if k in appraisal
        }},
    }
    return effect
```

- [ ] **Step 3: Wire spark worker `_append_turn_effect_metadata`**

Prefer `turn_effect_from_appraisal` before `turn_effect_from_spark_meta`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_turn_effect.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/telemetry/turn_effect.py tests/test_turn_effect.py services/orion-spark-introspector/app/worker.py
git commit -m "feat: turn effect helper from turn_change_appraisal"
```

---

### Task 11: Introspection prompt template

**Files:**
- Modify: `orion/cognition/prompts/introspect_spark.j2`

- [ ] **Step 1: Replace φ table with appraisal block**

```jinja2
=== TURN CHANGE APPRAISAL ===
{% set appr = metadata.spark_meta.turn_change_appraisal or {} %}
status={{ appr.get('turn_change_status', 'missing') }}
baseline={{ appr.get('baseline_mode', '?') }}
novelty_score={{ appr.get('novelty_score', 'null') }}
shift_kind={{ appr.get('shift_kind', '?') }}
confidence={{ appr.get('confidence', '?') }}

{% if metadata.spark_meta.self_state_v1 %}
=== ORGANISM CONDITION (SelfStateV1 snapshot) ===
{{ metadata.spark_meta.self_state_v1 }}
{% endif %}
```

Remove the `phi_before` / `phi_after` metric table (lines 11–36 in current template).

- [ ] **Step 2: Update analysis task copy**

Change task 1 to reference appraisal shift kind and novelty score instead of φ deltas.

- [ ] **Step 3: Commit**

```bash
git add orion/cognition/prompts/introspect_spark.j2
git commit -m "refactor: introspect spark template uses turn_change_appraisal"
```

---

### Task 12: End-to-end verification sweep

- [ ] **Step 1: Unit suite**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_turn_change_classify.py tests/test_consolidation_classify.py tests/test_turn_effect.py tests/test_substrate_signal_bridge.py -q`
Expected: all passed

- [ ] **Step 2: Service suite**

Run: `./scripts/test_service.sh orion-memory-consolidation -q`
Expected: all passed

- [ ] **Step 3: Compile touched services**

Run: `python -m compileall orion/memory services/orion-memory-consolidation/app services/orion-spark-introspector/app services/orion-llm-gateway/app`
Expected: exit 0

- [ ] **Step 4: Manual acceptance checklist (live stack — operator)**

1. Hub chat turn → `spark_meta.turn_change_appraisal` appears within patch timeout (~1s).
2. Spark telemetry `novelty` matches appraisal `novelty_score` when `turn_change_status=ok`.
3. New turns have no `phi_before`/`phi_after` from gateway.
4. Confident novel turn → bus message on `orion:signals:memory_consolidation` with kind `signal.memory_consolidation.turn_change`.

---

## Self-Review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| `turn_change_appraisal` in spark_meta patch | 5, 7 |
| Logprob-only happy path scores | 1, 3, 5 |
| Spark telemetry `novelty` from appraisal | 8 |
| Substrate `organ_signal` on threshold | 6, 7 |
| Degraded → `novelty=null`, no tissue fallback | 8, 10 |
| Gateway stops φ stamping | 9 |
| Env vars + `.env_example` sync | 4 |
| Prior turn + session window fallback | 5 |
| Tissue removed from hot path (no shadow) | 8, 9 |
| Classify prompt without `phi_after.novelty` | 2 |
| `introspect_spark.j2` appraisal input | 11 |
| Unit + integration tests | 1, 3, 5, 6, 7, 10, 12 |

**Known v1 limits (documented, not tasks):** async patch lag; classifier bias; SelfStateV1 poll lag; Appendix A (substrate context hints) deferred.

**Placeholder scan:** No TBD/TODO steps in this plan.

**Type consistency:** `scores_from_llm_result` returns dict; `classify_turn` nests under `turn_change_appraisal`; `build_turn_change_signal` uses same `shift_kind` strings as `SHIFT_KINDS`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-22-turn-change-appraisal-v1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
