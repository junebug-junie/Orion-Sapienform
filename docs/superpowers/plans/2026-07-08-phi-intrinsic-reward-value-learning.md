# φ Intrinsic Reward Value Learning (Step 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume `PhiIntrinsicRewardV1` from `orion:self:phi_reward`, persist durable trace rows, roll up episodes, and optionally feed bounded coherence drive pressure via `orion/autonomy/phi_value_metabolism.py` — all default-off except the recorder.

**Architecture:** `orion-substrate-runtime` subscribes to the bus, appends to `substrate_phi_value_trace`, materializes `substrate_phi_value_episode` on a cursor. `orion/spark/concept_induction/bus_worker.py` reads the latest episode and applies metabolism when `ORION_PHI_VALUE_LEARNING_ENABLED=true`. No drive mutation inside substrate-runtime.

**Tech Stack:** Python 3, Pydantic v2, pytest, Postgres (substrate store), Redis bus.

**Spec:** `docs/superpowers/specs/2026-07-08-phi-intrinsic-reward-value-learning-design.md`

**Prerequisite:** Plan 2 producer tests pass (`docs/superpowers/plans/2026-07-08-phi-encoder-plan2.md` Task 9+). Can develop against fake `PhiIntrinsicRewardV1` fixtures in parallel.

---

## File structure

**Created:**
- `services/orion-substrate-runtime/app/phi_value_recorder.py`
- `services/orion-substrate-runtime/app/phi_value_episode.py`
- `orion/autonomy/phi_value_metabolism.py`
- `services/orion-substrate-runtime/tests/test_phi_value_recorder.py`
- `services/orion-substrate-runtime/tests/test_phi_value_episode.py`
- `orion/autonomy/tests/test_phi_value_metabolism.py`
- `services/orion-substrate-runtime/evals/eval_phi_intrinsic_drive_feed.py`

**Modified:**
- `services/orion-substrate-runtime/app/worker.py` (or bus listener)
- `services/orion-substrate-runtime/app/store.py` (trace + episode tables)
- `services/orion-substrate-runtime/app/settings.py` + `.env_example`
- `orion/spark/concept_induction/bus_worker.py` (optional metabolism hook)

---

## Task 1: Trace table + recorder

**Files:**
- Modify: `services/orion-substrate-runtime/app/store.py`
- Create: `services/orion-substrate-runtime/app/phi_value_recorder.py`
- Create: `services/orion-substrate-runtime/tests/test_phi_value_recorder.py`

- [ ] **Step 1: Write failing persist test** (fake SQL engine pattern from hub tests)

```python
def test_phi_value_recorder_persists_ok_event(fake_engine) -> None:
    from app.phi_value_recorder import PhiValueRecorder
    from orion.schemas.telemetry.phi_encoder import PhiIntrinsicRewardV1
    from datetime import datetime, timezone

    rec = PhiValueRecorder(engine=fake_engine)
    event = PhiIntrinsicRewardV1(
        generated_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
        self_state_id="self.state:tick_1:policy.v1",
        encoder_version="v0",
        features_version="seed-v2",
        phi=0.7,
        delta_phi=-0.02,
        recon_error=0.1,
        delta_recon_error=0.01,
    )
    assert rec.persist(event) is True
    assert fake_engine.insert_count == 1
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement store methods + recorder**

`persist(event) -> bool`:
- Return False without insert if `phi_health != "ok"` or `grammar_truth_degraded`
- `INSERT ... ON CONFLICT (encoder_version, self_state_id) DO NOTHING` for idempotency

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

---

## Task 2: Bus subscribe + wire recorder

**Files:**
- Modify: `services/orion-substrate-runtime/app/worker.py`
- Modify: `services/orion-substrate-runtime/app/settings.py` + `.env_example`

- [ ] **Step 1: Add settings**

```python
channel_phi_reward: str = Field("orion:self:phi_reward", alias="CHANNEL_PHI_REWARD")
phi_value_recorder_enabled: bool = Field(True, alias="PHI_VALUE_RECORDER_ENABLED")
```

- [ ] **Step 2: Subscribe handler** — parse envelope payload to `PhiIntrinsicRewardV1`, call recorder; log errors, never crash worker.

- [ ] **Step 3: Integration test with fake bus message**

- [ ] **Step 4: Commit**

---

## Task 3: Episode reducer

**Files:**
- Create: `services/orion-substrate-runtime/app/phi_value_episode.py`
- Create: `services/orion-substrate-runtime/tests/test_phi_value_episode.py`

- [ ] **Step 1: Write failing aggregate test**

```python
def test_episode_reducer_sum_delta_phi() -> None:
    from app.phi_value_episode import reduce_episode
    rows = [
        {"delta_phi": -0.01, "phi": 0.7, "recon_error": 0.1, "encoder_version": "v0"},
        {"delta_phi": -0.02, "phi": 0.68, "recon_error": 0.12, "encoder_version": "v0"},
    ]
    ep = reduce_episode(rows)
    assert abs(ep.sum_delta_phi - (-0.03)) < 1e-9
    assert ep.tick_count == 2
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement reducer** — cursor over trace table, window `PHI_VALUE_EPISODE_WINDOW_SEC` (300), split on encoder_version change.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

---

## Task 4: phi_value_metabolism (drive feed)

**Files:**
- Create: `orion/autonomy/phi_value_metabolism.py`
- Create: `orion/autonomy/tests/test_phi_value_metabolism.py`

- [ ] **Step 1: Write failing tests**

```python
from orion.autonomy.phi_value_metabolism import coherence_impact_from_episode


def test_drive_feed_default_off_flag() -> None:
    assert coherence_impact_from_episode(
        {"sum_delta_phi": -0.2}, enabled=False,
    ) == 0.0


def test_drive_feed_negative_delta_raises_impact() -> None:
    impact = coherence_impact_from_episode(
        {"sum_delta_phi": -0.2},
        enabled=True,
        delta_eps=0.05,
        delta_scale=0.5,
        weight=0.15,
    )
    assert impact > 0.0
    assert impact <= 0.15


def test_drive_feed_flat_delta_inert() -> None:
    assert coherence_impact_from_episode(
        {"sum_delta_phi": 0.1}, enabled=True,
    ) == 0.0
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
def coherence_impact_from_episode(
    episode: dict,
    *,
    enabled: bool,
    delta_eps: float = 0.05,
    delta_scale: float = 0.5,
    weight: float = 0.15,
) -> float:
    if not enabled:
        return 0.0
    sum_delta = float(episode.get("sum_delta_phi") or 0.0)
    if sum_delta >= -delta_eps:
        return 0.0
    pressure = min(1.0, max(0.0, -sum_delta / delta_scale))
    return weight * pressure
```

- [ ] **Step 4: Wire into concept-induction bus_worker** behind `ORION_PHI_VALUE_LEARNING_ENABLED=false` default — read latest episode row, add impact to coherence drive input.

- [ ] **Step 5: Run tests — expect PASS**

- [ ] **Step 6: Commit**

---

## Task 5: Eval + gate

- [ ] **Step 1:** `eval_phi_intrinsic_drive_feed.py` — synthetic 1h trace, assert bounded impact.

- [ ] **Step 2: Run gate**

```bash
pytest services/orion-substrate-runtime/tests/test_phi_value_recorder.py \
  services/orion-substrate-runtime/tests/test_phi_value_episode.py \
  orion/autonomy/tests/test_phi_value_metabolism.py -q
python scripts/check_schema_registry.py
```

- [ ] **Step 3: Commit**

---

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
# concept-induction if bus_worker changed:
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

`ORION_PHI_VALUE_LEARNING_ENABLED` stays `false` until operator enables after trace validation.
