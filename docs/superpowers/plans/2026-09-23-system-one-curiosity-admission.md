# Implementation plan: System One curiosity admission

> **For agentic workers:** implement task-by-task; checkboxes track progress.

**Goal:** Give `curiosity_pull` categorical causal influence as an admission gate over endogenous curiosity, without promoting other System One questions.

**Architecture:** Reusable frame access → curiosity admission decision → wire into `_endogenous_curiosity_tick` before `FrontierCuriosityEvaluator`; persist gate lineage on candidate rows; publish typed frame on an operational bus channel; update eval script for post-live calibration.

**Tech Stack:** Python, Pydantic frames, substrate-runtime worker/store, Redis bus channel registry, pytest.

## Global Constraints

- No new state ontology / propensity soup / cross-question aggregates
- Argmax categorical: 0=noop, 1|2=admit (same effect)
- Fail-open when System One unavailable
- Kill switch default live (false)
- Do not merge the PR

---

### Task 1: Access + admission module + unit tests

- [x] `orion/substrate/system_one_access.py`
- [ ] `tests/test_system_one_access.py` / substrate tests

### Task 2: Store gate_json + migration

- [ ] Additive `gate_json` column
- [ ] `save_endogenous_curiosity_candidates(..., gate=...)`

### Task 3: Wire worker + settings + bus publish

- [ ] Curiosity tick gate
- [ ] Kill switch env
- [ ] `orion:system_one:appraisal` channel + publish

### Task 4: Eval script + docs + PR

- [ ] Post-live metrics in `eval_system_one_appraisal.py`
- [ ] Spec/README updates (not universally shadow-only)
- [ ] Open PR, do not merge
