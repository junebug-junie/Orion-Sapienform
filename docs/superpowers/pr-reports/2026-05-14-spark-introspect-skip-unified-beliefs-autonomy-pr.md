# PR: Spark introspection — skip unified beliefs & autonomy GraphDB

## Branch

- **Head:** `feat/spark-introspect-skip-unified-beliefs-autonomy`
- **Base:** Prefer **`main`** (or your integration branch). This branch was created from **`feat/hub-deterministic-skill-runner-and-stash-tooling`** at the time of the commit; rebase onto `main` before merge if that parent is not the desired base.

Create PR: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/spark-introspect-skip-unified-beliefs-autonomy

---

## Summary

`introspect_spark` was still paying **Cognitive Unification** and **`autonomy_ctx_adapter`** cost because **`prepare_brain_reply_context` ran from `PlanRunner`** (router) for brain-mode verbs, independent of the executor `_should_prepare_brain_reply_context` gate. This change adds **adapter- and layer-level hard skips**, **router skip** for brain prep on spark introspection, **explicit cortex `options` flags** from the spark-introspector worker, and **tests** so GraphDB / `list_latest` autonomy fan-out does not run on the spark introspection path.

---

## What changed

| Area | Change |
|------|--------|
| **`orion/substrate/relational/adapters/autonomy_ctx.py`** | `map_autonomy_ctx_to_substrate`: early return for `introspect_spark`, `execution_lane`/`llm_lane` == `spark`, `skip_unified_beliefs` / `skip_autonomy_context` (ctx or `options`). |
| **`orion/substrate/relational/layer.py`** | `_skip_unified_beliefs_ctx`, `_lightweight_belief_set`; `beliefs_for_stance` returns empty slices + lineage marker before any producer fan-out. |
| **`services/orion-cortex-exec/app/chat_stance.py`** | `_unified_beliefs_for_stance` short-circuits using layer skip helpers before `_get_unification_layer()`. |
| **`services/orion-cortex-exec/app/router.py`** | Skip `prepare_brain_reply_context` when `skip_brain_reply_context` or `verb == introspect_spark`. |
| **`services/orion-cortex-exec/app/executor.py`** | `_should_prepare_brain_reply_context`: skip `introspect_spark` (brain prep per exec step). |
| **`services/orion-cortex-exec/tests/test_executor_runtime_context_skip.py`** | Assert introspect spark skips brain reply context prep. |
| **`services/orion-spark-introspector/app/worker.py`** | Cortex request `options`: `skip_brain_reply_context`, `skip_unified_beliefs`, `skip_autonomy_context`, `skip_chat_stance_inputs`. |
| **`orion/substrate/relational/tests/test_adapters.py`** | `TestAutonomyCtxAdapterSkip`. |
| **`orion/substrate/relational/tests/test_layer.py`** | `test_beliefs_for_stance_skips_producers_for_introspect_spark`. |

---

## Verification (ran locally)

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./venv/bin/python -m pytest \
  orion/substrate/relational/tests/test_adapters.py \
  orion/substrate/relational/tests/test_layer.py \
  orion/substrate/relational/tests/test_golden_path.py \
  services/orion-cortex-exec/tests/test_executor_runtime_context_skip.py \
  services/orion-cortex-exec/tests/test_chat_stance_brief.py \
  -q --tb=short
# 77 passed (subset run during development); re-run full suite if your CI scope differs.
```

---

## Risk / follow-up

- **`execution_lane == "spark"`** in `_skip_unified_beliefs_ctx` skips unified beliefs for **any** verb on that lane, not only `introspect_spark`. If a future verb needs full beliefs on the spark exec lane, tighten the predicate (e.g. require `verb == "introspect_spark"` in addition to lane).

---

## Suggested PR title

**fix(spark-introspect): skip unified beliefs and autonomy GraphDB on spark path**

---

## Suggested PR description (copy-paste)

### What & why

Spark heavy introspection (`introspect_spark`) only needs `metadata` for `introspect_spark.j2`, but **router-level `prepare_brain_reply_context`** still invoked **`build_chat_stance_inputs` → unified layer → `map_autonomy_ctx_to_substrate`**, producing duplicate autonomy consumers (`chat_stance` + `autonomy_ctx_adapter`) and GraphDB fan-out. This PR blocks that at the **router**, **unification layer**, **autonomy adapter**, **chat_stance** early exit, **executor** brain-prep gate, and **spark-introspector** cortex `options`.

### Key behavior

- **Autonomy adapter:** no repository / `list_latest` when spark introspection, spark lane, or skip flags are set.
- **CognitiveUnificationLayer:** returns a lightweight `UnifiedRelationalBeliefSetV1` with lineage `skipped:introspect_spark_or_unified_beliefs_disabled` without running producers.
- **PlanRunner:** does not call `prepare_brain_reply_context` for `introspect_spark` or when `skip_brain_reply_context` is true after options merge.
- **Spark introspector:** sets explicit `skip_*` options on the cortex request for downstream consistency.

### How to test

```bash
PYTHONPATH=. ./venv/bin/python -m pytest orion/substrate/relational/tests/test_adapters.py \
  orion/substrate/relational/tests/test_layer.py orion/substrate/relational/tests/test_golden_path.py \
  services/orion-cortex-exec/tests/test_executor_runtime_context_skip.py \
  services/orion-cortex-exec/tests/test_chat_stance_brief.py -q --tb=short
```

### Checklist

- [ ] Rebase onto correct base (`main` vs parent feature branch) before merge.
- [ ] Confirm no other verb legitimately needs full unified beliefs on `execution_lane=spark`.
