# Collapse Mirror — ORION-lane φ-gated causal density — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ground the collapse-mirror `causal_density.score` for Orion-authored (metacog-lane) entries in computed `SelfStateV1`/φ evidence, instead of scoring purely from the entry's own self-reported numeric fields (which were typed by the same LLM turn that authored the entry's narrative). Leave Juniper/system-authored ("strict" lane) entries byte-for-byte unchanged.

**Architecture:** `score_causal_density()` (`orion/collapse/service.py`) currently computes `causal_density.score` as a mean of self-reported signals: `numeric_sisters` (valence/arousal/clarity/overload/risk_score — all self-typed), `constraints.severity_score`, `tag_scores`, `change_type_scores`. None of this is grounded in an independently-computed signal. `orion/schemas/collapse_mirror.py::mirror_kind()` already distinguishes three lanes: `"strict"` (Juniper-observed, or `source_service`/origin normalizes to `collapse_mirror_service`), `"metacog"` (Orion-observed, or origin service `metacog`), `"unknown"`. This plan gates the *scoring function only* — not the narrative fields, not entry creation/enrichment — so that when `mirror_kind(entry) == "metacog"`, the score blends in computed φ evidence (self_state prediction-error magnitude near the entry's timestamp, `overall_condition` severity transitions, `dimension_trajectory` swings), weighted against the self-reported signals. `"strict"` and `"unknown"` lanes keep the exact current behavior.

**Tech Stack:** Python 3.12, pydantic v2 (`CollapseMirrorEntryV2`, `SelfStateV1`), plain in-process JSON file store (`CollapseMirrorStore`, no DB/bus dependency today), `orion-cortex-exec` verb layer (`BaseVerb`/`VerbContext`).

**Design spec:** No companion spec doc — the change is scoped to one function's internal blend logic plus a data-access decision that this plan surfaces as an open question rather than resolving unilaterally (see Open Questions). Architecture rationale is carried inline here.

**Parent plan:** None.

**Worktree:** Implement in an isolated worktree (`using-superpowers:using-git-worktrees`) before touching main — this changes a scoring function invoked from a live cortex-exec verb (`ScoreCausalDensityVerb`) during real chat turns.

---

## Verified findings (read before implementing)

1. **`mirror_kind()` lane logic** (`orion/schemas/collapse_mirror.py`, lines 267–291): returns `"strict"` if `observer` normalizes to `"juniper"`, OR if `origin_service` normalizes to `"collapse_mirror_service"`. Returns `"metacog"` if `observer` normalizes to `"orion"`, OR if `origin_service` normalizes to `"metacog"`. Else `"unknown"`. This is the existing, load-bearing lane distinction — this plan gates strictly on `mirror_kind(entry) == "metacog"`.

2. **`score_causal_density()` today** (`orion/collapse/service.py`, lines 132–177): signature is `score_causal_density(event_id: str) -> CollapseMirrorEntryV2`. It fetches the entry from the store, builds a `signals: list[float]` from `abs(numeric.valence/arousal/clarity/overload/risk_score)`, `constraints.severity_score`, `max(tag_scores.values())`, `max(change_type_scores.values())`, averages via `_score_from_values` (mean, clamped `[0,1]`), labels via `_label_for_score` (`critical>=.85`, `dense>=.6`, `salient>=.25`, else `ambient`), and sets `entry.is_causally_dense = score >= 0.6`. All of these inputs are self-reported at entry-authoring time — there is no independently-computed signal anywhere in this function today.

3. **`CollapseMirrorStore` is a plain local JSON file, not Postgres-backed** (`orion/collapse/service.py`, lines 19–80). `DEFAULT_STORE_PATH = "/mnt/storage/collapse-mirrors/collapse_mirror_store.json"`, overridable via `COLLAPSE_MIRROR_STORE_PATH` env. It has **zero existing DB or bus dependency** — `orion/collapse/service.py` imports nothing from `orion.core.bus` or any SQLAlchemy engine. This is an explicit architectural fork point: adding a self_state read here means choosing between (a) giving this module its own Postgres connection against self-state-runtime's tables, (b) a bus/RPC call to `orion-state-service` or `orion-self-state-runtime`, or (c) keeping `score_causal_density()` itself pure/injectable and having the *caller* (the verb layer) fetch self_state and pass it in. See Open Questions — this plan recommends (c) but does not treat it as settled.

4. **The call site is `ScoreCausalDensityVerb.execute()`** (`services/orion-cortex-exec/app/collapse_verbs.py`, lines 125–151 approx.): `entry = score_causal_density(payload.event_id)`, where `payload: CollapseMirrorEventRequest` has only `event_id: str`. `ctx.meta` carries `bus` and `source`. This is where a self_state fetch would need to be injected under option (c).

5. **A directly reusable self_state read pattern already exists in the same service** — `services/orion-cortex-exec/app/substrate_felt_state_reader.py::SubstrateFeltStateReader` already does a direct Postgres read of the `substrate_self_state` table (`payload_col=self_state_json`, `ts_col=generated_at`), with a freshness window (`SUBSTRATE_FELT_STATE_MAX_AGE_SEC`, default 120s) and an in-process cache, gated by `ENABLE_SUBSTRATE_FELT_STATE_CTX` (already `true` by default in `orion-cortex-exec/.env_example`). This is the same table `SelfStateRuntimeStore.save_self_state` (`services/orion-self-state-runtime/app/store.py`) writes to. **This is strong precedent for option (c):** the verb layer already has a working, tested, freshness-gated direct-Postgres self_state reader in-process; reusing/extending it (rather than adding a second bus/RPC path, or building a fresh Postgres connection inside `orion/collapse/service.py`) is the thinnest available seam. It is presented here as the strongest candidate, not a unilateral decision — see Open Questions.

6. **`orion-state-service` (`services/orion-state-service/README.md`) does not currently consume self_state at all.** Its documented inputs are `spark.state.snapshot.v1` and biometrics summaries/inductions/cluster; its only RPC is `state.get_latest.v1` → `state.latest.reply.v1`. Extending it to also cache `substrate.self_state.v1` (the broadcast published by self-state-runtime; see the companion pacemaker plan, `docs/superpowers/plans/2026-07-01-orion-heartbeat-pacemaker-v1.md`, for confirmation this broadcast already exists and — after that plan's Phase 1 — fires continuously) is a real, viable alternative to option (c) above, reusing this service's existing "latest snapshot oracle" role. It is *not* implemented here; it is listed in Open Questions as an alternative to weigh against reusing `substrate_felt_state_reader.py`.

7. **Timestamp alignment field:** `CollapseMirrorEntryV2.timestamp` (`orion/schemas/collapse_mirror.py`, line 474) is `Optional[str]`, defaulting to `_utc_now_iso()` at construction. This is the field to align against `SelfStateV1.generated_at` when locating "the self_state nearest the entry's timestamp."

8. **Narrative fields are untouched by this plan:** `summary`, `mantra`, `trigger`, `causal_echo`, `emergent_entity` remain fully Orion-authored language. Only `causal_density.score` (and, per Open Questions, possibly `is_causally_dense`) is affected.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `orion/collapse/service.py` | Modify | Add φ-blended scoring for the metacog lane; strict/unknown lanes byte-for-byte unchanged |
| `services/orion-cortex-exec/app/collapse_verbs.py` | Modify (only if option (c) chosen — see Open Questions) | Fetch current self_state, pass into scoring call |
| `tests/test_collapse_service_causal_density.py` | Create | Golden-fixture tests: strict-lane parity, metacog pull-down, metacog pull-up |

---

## Task 1: φ-gated causal density for the ORION (metacog) lane only

**Files:**
- Modify: `orion/collapse/service.py`
- Modify (conditionally, per Open Question 1): `services/orion-cortex-exec/app/collapse_verbs.py`
- Create: `tests/test_collapse_service_causal_density.py`

**Design decision this task assumes (revisit if Open Question 1 resolves differently):** keep `score_causal_density(event_id: str)` as the existing pure/no-new-dependency entry point used by strict/unknown lanes and by any other current caller, and add a new function `score_causal_density_with_self_state(event_id: str, self_state: SelfStateV1 | dict | None) -> CollapseMirrorEntryV2` that `ScoreCausalDensityVerb` calls instead. This keeps `orion/collapse/service.py` free of any new DB/bus dependency (preserving finding #3) and keeps the blend logic unit-testable with a plain in-memory `SelfStateV1`/dict fixture, with no network or Postgres mock required. `score_causal_density()` itself becomes a one-line delegation: `return score_causal_density_with_self_state(event_id, self_state=None)`, so every existing call site (including the strict lane) keeps working unchanged and self_state defaults to absent (which must produce identical output to today — verified by Task 1's Step 1(a) test).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_collapse_service_causal_density.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion.collapse.service import (
    CollapseMirrorStore,
    create_entry_from_v2,
    score_causal_density_with_self_state,
)
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1


def _store(tmp_path: Path) -> CollapseMirrorStore:
    return CollapseMirrorStore(str(tmp_path / "collapse_mirror_store.json"))


def _base_entry_payload(*, observer: str, source_service: str | None, numeric_overrides: dict | None = None) -> dict:
    payload = {
        "observer": observer,
        "trigger": "test trigger",
        "type": "test",
        "emergent_entity": "test entity",
        "summary": "test summary",
        "mantra": "test mantra",
        "numeric_sisters": {
            "valence": 0.1,
            "arousal": 0.1,
            "clarity": 0.1,
            "overload": 0.1,
            "risk_score": 0.1,
        },
    }
    if numeric_overrides:
        payload["numeric_sisters"].update(numeric_overrides)
    if source_service:
        payload["source_service"] = source_service
    return payload


def _steady_self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="ss-1",
        generated_at="2026-07-01T00:00:00+00:00",
        source_field_tick_id="tick-1",
        source_field_generated_at="2026-07-01T00:00:00+00:00",
        source_attention_frame_id="frame-1",
        source_attention_generated_at="2026-07-01T00:00:00+00:00",
        overall_condition="steady",
        overall_intensity=0.3,
        overall_confidence=0.8,
        dimensions={"execution_pressure": SelfStateDimensionV1(dimension_id="execution_pressure", score=0.2, confidence=0.8)},
        prediction_error_scores={"execution_pressure": 0.02},
        trajectory_condition="stable",
    )


def _unstable_self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="ss-2",
        generated_at="2026-07-01T00:05:00+00:00",
        source_field_tick_id="tick-2",
        source_field_generated_at="2026-07-01T00:05:00+00:00",
        source_attention_frame_id="frame-2",
        source_attention_generated_at="2026-07-01T00:05:00+00:00",
        overall_condition="unstable",
        overall_intensity=0.9,
        overall_confidence=0.7,
        dimensions={"execution_pressure": SelfStateDimensionV1(dimension_id="execution_pressure", score=0.85, confidence=0.7)},
        prediction_error_scores={"execution_pressure": 0.62},
        trajectory_condition="degrading",
    )


def test_strict_lane_score_unchanged_with_or_without_self_state(tmp_path, monkeypatch):
    import orion.collapse.service as svc
    monkeypatch.setattr(svc, "_get_store", lambda: _store(tmp_path))

    entry = create_entry_from_v2(
        _base_entry_payload(observer="Juniper", source_service=None, numeric_overrides={"risk_score": 0.9}),
    )

    without_self_state = score_causal_density_with_self_state(entry.event_id, self_state=None)
    score_a = without_self_state.causal_density.score

    with_self_state = score_causal_density_with_self_state(entry.event_id, self_state=_unstable_self_state())
    score_b = with_self_state.causal_density.score

    assert score_a == score_b, "strict-lane entries must not be affected by self_state at all"


def test_metacog_lane_high_self_report_but_steady_self_state_pulls_score_down(tmp_path, monkeypatch):
    import orion.collapse.service as svc
    monkeypatch.setattr(svc, "_get_store", lambda: _store(tmp_path))

    entry = create_entry_from_v2(
        _base_entry_payload(
            observer="Orion",
            source_service="metacog",
            numeric_overrides={"valence": 0.95, "arousal": 0.95, "risk_score": 0.95},
        ),
    )
    self_report_only = score_causal_density_with_self_state(entry.event_id, self_state=None)
    blended = score_causal_density_with_self_state(entry.event_id, self_state=_steady_self_state())

    assert blended.causal_density.score < self_report_only.causal_density.score
    assert blended.causal_density.label in {"salient", "ambient"}


def test_metacog_lane_modest_self_report_but_severe_self_state_pulls_score_up(tmp_path, monkeypatch):
    import orion.collapse.service as svc
    monkeypatch.setattr(svc, "_get_store", lambda: _store(tmp_path))

    entry = create_entry_from_v2(
        _base_entry_payload(
            observer="Orion",
            source_service="metacog",
            numeric_overrides={"valence": 0.2, "arousal": 0.2, "risk_score": 0.2},
        ),
    )
    self_report_only = score_causal_density_with_self_state(entry.event_id, self_state=None)
    blended = score_causal_density_with_self_state(entry.event_id, self_state=_unstable_self_state())

    assert blended.causal_density.score > self_report_only.causal_density.score
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_collapse_service_causal_density.py -v`
Expected: FAIL (`ImportError: cannot import name 'score_causal_density_with_self_state'`)

- [ ] **Step 3: Implement the blend**

Add to `orion/collapse/service.py`, replacing the current `score_causal_density` (lines 132–177):

```python
from orion.schemas.self_state import SelfStateV1

# Named constants so the blend weighting is easy to find and tune later —
# not magic numbers buried in the function body. Starting defaults; see the
# plan's Open Questions for the tuning discussion.
METACOG_SELF_REPORT_WEIGHT = 0.35
METACOG_PHI_EVIDENCE_WEIGHT = 0.65

_SEVERITY_ORDER = ("quiet", "steady", "loaded", "strained", "unstable")


def _condition_severity_rank(condition: str | None) -> int:
    try:
        return _SEVERITY_ORDER.index(condition or "")
    except ValueError:
        return -1  # "unknown" or missing: not a signal either way


def _coerce_self_state(raw: Any) -> SelfStateV1 | None:
    if raw is None:
        return None
    try:
        if isinstance(raw, SelfStateV1):
            return raw
        if isinstance(raw, dict):
            return SelfStateV1.model_validate(raw)
        if isinstance(raw, str) and raw.strip():
            return SelfStateV1.model_validate_json(raw)
    except Exception as exc:
        logger.debug("collapse_density_self_state_parse_failed error=%s", exc)
    return None


def _phi_evidence_score(self_state: SelfStateV1 | None) -> float | None:
    """Computed φ evidence score in [0, 1], or None if no self_state available.

    Blends: (a) magnitude of prediction_error_scores (max, matching the
    existing overall-surprise convention used elsewhere in the self-model —
    see self_state_ctx.py), (b) overall_condition severity rank normalized to
    [0, 1], (c) whether trajectory_condition is "degrading" (a fixed bump,
    since a transition matters independent of absolute severity).
    """
    if self_state is None:
        return None
    prediction_error = max(
        (float(v or 0.0) for v in (self_state.prediction_error_scores or {}).values()),
        default=0.0,
    )
    severity_rank = _condition_severity_rank(self_state.overall_condition)
    severity_norm = max(0.0, severity_rank / (len(_SEVERITY_ORDER) - 1)) if severity_rank >= 0 else 0.0
    degrading_bump = 0.15 if self_state.trajectory_condition == "degrading" else 0.0
    evidence = max(0.0, min(1.0, 0.5 * prediction_error + 0.5 * severity_norm + degrading_bump))
    return evidence


def _self_report_signals(entry: CollapseMirrorEntryV2) -> list[float]:
    numeric = entry.numeric_sisters
    signals: list[float] = []
    for value in (numeric.valence, numeric.arousal, numeric.clarity, numeric.overload, numeric.risk_score):
        if value is None:
            continue
        try:
            signals.append(abs(float(value)))
        except (TypeError, ValueError):
            continue
    if numeric.constraints.severity_score is not None:
        try:
            signals.append(float(numeric.constraints.severity_score))
        except (TypeError, ValueError):
            pass
    if entry.tag_scores:
        tag_values = [float(v) for v in entry.tag_scores.values() if v is not None]
        if tag_values:
            signals.append(max(tag_values))
    if entry.change_type_scores:
        change_values = [float(v) for v in entry.change_type_scores.values() if v is not None]
        if change_values:
            signals.append(max(change_values))
    return signals


def score_causal_density(event_id: str) -> CollapseMirrorEntryV2:
    """Unchanged public entry point — no self_state, no behavior change for
    any existing caller. Strict-lane and unknown-lane entries always go
    through this path with self_state=None, which is byte-for-byte identical
    to pre-this-change behavior."""
    return score_causal_density_with_self_state(event_id, self_state=None)


def score_causal_density_with_self_state(
    event_id: str,
    self_state: SelfStateV1 | dict | str | None,
) -> CollapseMirrorEntryV2:
    store = _get_store()
    entry = store.get(event_id)

    self_report_signals = _self_report_signals(entry)
    self_report_score = _score_from_values(self_report_signals)

    lane = mirror_kind(entry)
    if lane == "metacog":
        phi_score = _phi_evidence_score(_coerce_self_state(self_state))
        if phi_score is None:
            # No self_state available this call — fall back to pure self-report,
            # identical to strict-lane behavior, rather than silently blending
            # with a fabricated zero.
            score = self_report_score
        else:
            score = max(
                0.0,
                min(
                    1.0,
                    METACOG_SELF_REPORT_WEIGHT * self_report_score
                    + METACOG_PHI_EVIDENCE_WEIGHT * phi_score,
                ),
            )
    else:
        # strict / unknown: exactly the pre-existing behavior, no self_state involved.
        score = self_report_score

    label = _label_for_score(score)

    entry.causal_density = CollapseMirrorCausalDensity(
        label=label,
        score=score,
        rationale=entry.causal_density.rationale or "computed_from_numeric_sisters",
    )

    entry.is_causally_dense = score >= 0.6
    if entry.snapshot_kind == "baseline" and entry.is_causally_dense:
        entry.snapshot_kind = "confirmed_dense"

    store.save(entry)
    return entry
```

Add `mirror_kind` to the existing import from `orion.schemas.collapse_mirror` at the top of the file:
```python
from orion.schemas.collapse_mirror import (
    CollapseMirrorCausalDensity,
    CollapseMirrorEntryV2,
    mirror_kind,
    normalize_collapse_entry,
)
```

**Note on the "unchanged" claim for strict lane:** `score_causal_density()` now delegates to `score_causal_density_with_self_state(event_id, self_state=None)`, and the `lane == "metacog"` branch is only taken when `mirror_kind(entry) == "metacog"` AND a non-`None` `self_state` is actually supplied. For strict-lane entries the `else` branch runs unconditionally and computes `self_report_score` identically to the pre-change function — same `_self_report_signals` extraction, same `_score_from_values` call, same `_label_for_score` call. This is what Step 1's `test_strict_lane_score_unchanged_with_or_without_self_state` verifies directly (passing `self_state=_unstable_self_state()` to a strict-lane entry must not move the score at all, because the lane check gates the phi branch before self_state is ever consulted).

- [ ] **Step 4: Wire the verb layer to pass self_state (only if Open Question 1 resolves to option (c) — reuse `substrate_felt_state_reader`)**

In `services/orion-cortex-exec/app/collapse_verbs.py::ScoreCausalDensityVerb.execute()`, replace:
```python
entry = score_causal_density(payload.event_id)
```
with:
```python
from app.substrate_felt_state_reader import hydrate_felt_state_ctx
from orion.collapse import score_causal_density_with_self_state

felt_state_ctx: Dict[str, Any] = {}
hydrate_felt_state_ctx(felt_state_ctx)
entry = score_causal_density_with_self_state(payload.event_id, self_state=felt_state_ctx.get("self_state"))
```
and export `score_causal_density_with_self_state` from `orion/collapse/__init__.py` alongside the existing `score_causal_density` export. **Do not implement this step until Open Question 1 is explicitly resolved** — if the follow-up call picks option (a) or (b) instead, this step's shape changes (a Postgres read local to `orion/collapse/service.py`, or a bus RPC call, respectively).

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_collapse_service_causal_density.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run existing collapse-adjacent suites for regressions**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_journaler_worker.py services/orion-actions/tests/test_journal_actions.py services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py -v`
Expected: PASS (confirms nothing importing `orion.collapse` broke)

- [ ] **Step 7: Commit**

```bash
git add orion/collapse/service.py orion/collapse/__init__.py tests/test_collapse_service_causal_density.py
git commit -m "feat: blend computed self_state evidence into metacog-lane causal density"
```
(Add `services/orion-cortex-exec/app/collapse_verbs.py` to this commit only if Step 4 was implemented.)

---

## Non-goals

- No change to the narrative fields (`summary`, `mantra`, `trigger`, `causal_echo`, `emergent_entity`) — those remain fully Orion-authored language. Only `causal_density.score` (and see Open Question 3 for `is_causally_dense`) is gated.
- No change to `LogCollapseMirrorVerb` or `EnrichCollapseMirrorVerb`.
- No change to strict-lane or unknown-lane behavior at all — verified by a golden-fixture test, not just asserted.
- No new bus channel, no new Postgres table, no schema migration.

---

## Open Questions (required — resolve or explicitly punt before implementation)

1. **Self_state data-access mechanism.** Three options, unresolved:
   - **(a) Direct Postgres read** from `orion/collapse/service.py` against self-state-runtime's `substrate_self_state` table. Pro: no new cross-service call, symmetric with how self-state-runtime persists. Con: gives a previously DB-free module a new hard dependency and its own connection-pool lifecycle to manage.
   - **(b) Bus/HTTP RPC** to `orion-state-service` or `orion-self-state-runtime`. Pro: decouples `orion/collapse/service.py` from Postgres entirely; fits `orion-state-service`'s existing "latest snapshot oracle" role if it's extended to cache `substrate.self_state.v1` (see Verified Finding #6). Con: `orion-state-service` doesn't consume self_state today — this is new scope on that service, and adds a network round-trip to a scoring call that is otherwise synchronous and local.
   - **(c) Injected callable/provider** — keep `score_causal_density_with_self_state()` pure (accepts `self_state` as a parameter, does no I/O itself), and have the verb-layer caller (`ScoreCausalDensityVerb` in `collapse_verbs.py`) fetch self_state and pass it in, reusing the already-existing `SubstrateFeltStateReader`/`hydrate_felt_state_ctx` pattern already living in the same service (`services/orion-cortex-exec/app/substrate_felt_state_reader.py`). Pro: keeps the scoring function itself deterministic and trivially unit-testable (Task 1's tests need no DB/bus mock at all); reuses a pattern that already exists, is already tested, and is already freshness-gated. Con: couples the *caller* (verb layer) to fetching self_state, meaning any other future caller of `score_causal_density_with_self_state` for a metacog-lane entry must also remember to fetch and pass self_state, or it silently falls back to pure self-report (which is safe but silently loses the intended grounding).

   **This plan's Task 1 is written assuming (c)**, because it is the thinnest seam and reuses an already-proven pattern in the same service, consistent with the repo's "reducers/scoring functions should be deterministic" convention seen elsewhere (e.g. `orion/self_state/prediction.py`'s pure functions). **This is a recommendation, not a decision** — flag for a human call before Task 1 Step 4 is implemented.

2. **Blend weighting.** `METACOG_SELF_REPORT_WEIGHT = 0.35` / `METACOG_PHI_EVIDENCE_WEIGHT = 0.65` in Task 1 are a **starting default, not a tuned value.** They were chosen to make computed φ evidence dominant (since the self-report side is the exact signal this plan doesn't trust for the metacog lane) while not zeroing out self-report entirely (per the task's explicit "blend, don't wholly replace" instruction). Revisit after this ships and a few weeks of metacog-lane entries can be eyeballed against their φ context.

3. **Scope of the gate: `score` only, or also `is_causally_dense`?** Task 1 as written applies the blended `score` to *both* the continuous `causal_density.score` and the boolean `is_causally_dense = score >= 0.6` threshold (since `is_causally_dense` is derived directly from `score` in the existing code and this plan does not introduce a second, independently-gated threshold). If a future reviewer wants `is_causally_dense` to remain self-report-only (e.g. because downstream consumers of that boolean have different risk tolerance than consumers of the continuous score), that would require decoupling the two, which this plan does not do. Flagged here as explicitly unresolved.

---

## Self-Review (spec coverage)

| Requirement | Task |
|---|---|
| Metacog lane blends computed φ evidence into `causal_density.score` | 1 |
| Strict lane byte-for-byte unchanged | 1 (test: `test_strict_lane_score_unchanged_with_or_without_self_state`) |
| Unknown lane unchanged | 1 (else branch covers both strict and unknown identically) |
| Self-report weighted down relative to φ when both available | 1 (named constants `METACOG_SELF_REPORT_WEIGHT`/`METACOG_PHI_EVIDENCE_WEIGHT`) |
| High self-report + flat self_state pulls score down | 1 (test: `test_metacog_lane_high_self_report_but_steady_self_state_pulls_score_down`) |
| Modest self-report + severe self_state/transition pulls score up | 1 (test: `test_metacog_lane_modest_self_report_but_severe_self_state_pulls_score_up`) |
| Narrative fields untouched | Non-goals |
| Data-access mechanism open question | Open Question 1 |
| Blend weighting open question | Open Question 2 |
| `is_causally_dense` scope open question | Open Question 3 |

**Known v1 limits (documented, not tasks):** Task 1 Step 4 (verb wiring) is explicitly gated on Open Question 1 being resolved and should not be implemented blind; the φ-evidence formula in `_phi_evidence_score` (prediction-error max + severity rank + degrading bump) is a first-pass heuristic, not derived from data, and is expected to be retuned once real metacog-lane entries can be compared against their self_state context.

**Placeholder scan:** No TBD/TODO steps. Task 1 Step 4 is conditionally gated (explicit condition stated), not a placeholder.
