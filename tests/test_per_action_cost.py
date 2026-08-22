"""The cost denominator, and the two places it used to be dropped.

A budget divides value by cost. Until 2026-08-21 Orion had no per-action cost
at all: `latency_ms` was a schema field, a database column and a reader, and
was populated on **0 of 5,739 rows over 6 hours**, because it was dropped
twice in series -- never written by the dispatch worker, and filtered out of
the evidence dict by the feedback store's four hardcoded keys.

These tests exist so it cannot silently become unreachable again. Both are
regression tests for a specific dead path, not for arithmetic.
"""

from __future__ import annotations

import inspect

from orion.schemas.action_prediction import ActionOutcomeRecordV1


class TestTheCostIsMeasuredAtTheSend:
    def test_the_dispatch_worker_times_the_cortex_call(self):
        """Measured around the send, not read from the verb's own report.

        Right quantity: wall-clock the action occupied on the motor path,
        queueing and transport included. Reliable: independent of whether a
        verb reports anything, and skills.runtime.* verbs report nothing.
        """
        import app.worker as worker  # type: ignore

        src = inspect.getsource(worker)
        assert "send_started = perf_counter()" in src
        # Every exit path from the send must carry a cost, including failure.
        assert src.count("perf_counter() - send_started") >= 3, (
            "a send that failed still consumed real time -- often the whole "
            "rpc timeout, the most expensive outcome there is. Recording it as "
            "absent makes failure look free."
        )

    def test_the_store_persists_it(self):
        import app.store as store  # type: ignore

        sig = inspect.signature(store.ExecutionDispatchRuntimeStore.save_dispatch_result)
        assert "latency_ms" in sig.parameters
        src = inspect.getsource(store.ExecutionDispatchRuntimeStore.save_dispatch_result)
        assert ":latency_ms" in src, "measured, accepted, and then not inserted"

    def test_every_bind_parameter_in_the_insert_is_actually_supplied(self):
        """Caught live, the hard way. `:latency_ms` was added to the INSERT and
        not to the params dict, so SQLAlchemy raised `A value is required for
        bind parameter 'latency_ms'` and dispatch-result writes started
        FAILING -- a regression shipped to a running service, found only by
        checking whether the number actually landed rather than trusting that
        the deploy succeeded."""
        import re

        import app.store as store  # type: ignore

        src = inspect.getsource(store.ExecutionDispatchRuntimeStore.save_dispatch_result)
        binds = set(re.findall(r"(?<!:):(\w+)(?![\w:])", src.split('"""')[1]))
        supplied = set(re.findall(r'"(\w+)":', src))
        missing = binds - supplied
        assert not missing, f"bind parameters with no supplied value: {sorted(missing)}"


class TestTheCostSurvivesToTheLedger:
    def test_the_feedback_evidence_dict_carries_latency(self):
        """The second drop. `_latencies()` scans these entries for
        latency_ms/duration_ms/elapsed_ms; the dict used to be exactly four
        hardcoded keys that excluded all three, which made the reader
        unreachable no matter what the producer wrote."""
        import app.store as fstore  # type: ignore

        src = inspect.getsource(fstore.FeedbackRuntimeStore.load_cortex_result_evidence)
        assert "latency_ms" in src
        assert 'entry["latency_ms"]' in src

    def test_absent_cost_stays_absent(self):
        """Never coerced to 0.0, which would read as 'this action was free'
        and bias any cost-weighted comparison toward whichever executor
        happens not to report timings."""
        import app.store as fstore  # type: ignore

        src = inspect.getsource(fstore.FeedbackRuntimeStore.load_cortex_result_evidence)
        assert 'row.get("latency_ms")' in src, (
            "must be .get() -- a mapping without the key is 'no cost "
            "recorded', not a KeyError that aborts the entire evidence load"
        )
        assert "if latency is not None" in src

    def test_the_ledger_field_still_permits_absence(self):
        assert ActionOutcomeRecordV1.model_fields["latency_ms"].default is None
