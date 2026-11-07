"""Phase G — compaction applier (memory mutation, hard-gated).

Every safety invariant is exercised against an in-memory fake store: default-off,
consent-gated (no consent gate exists yet, so a missing one fails closed; the policy-rung
frame it replaced retired 2026-11-07), snapshot-precedes-apply, downscale-first /
prune-gated, rollback-on-error, never-raises.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.compaction import (
    ConsolidateEntryV1,
    DownscaleEntryV1,
    MemoryCompactionDeltaV1,
    PruneEntryV1,
)

NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)


class FakeStore:
    """Records call order; models a rollback-able before/after state."""

    def __init__(self, *, fail_on: str | None = None, fail_snapshot: bool = False):
        self.calls: list[str] = []
        self.weights = {"e-1": 0.9, "e-2": 0.8}
        self.episodics = {"ep-1", "ep-2"}
        self.cards: list[str] = []
        self._fail_on = fail_on
        self._fail_snapshot = fail_snapshot

    def snapshot(self, delta):
        self.calls.append("snapshot")
        if self._fail_snapshot:
            raise RuntimeError("snapshot boom")
        return {"weights": dict(self.weights), "episodics": set(self.episodics), "cards": list(self.cards)}

    def restore(self, snapshot):
        self.calls.append("restore")
        self.weights = dict(snapshot["weights"])
        self.episodics = set(snapshot["episodics"])
        self.cards = list(snapshot["cards"])

    def downscale(self, target_id, old_w, new_w):
        self.calls.append(f"downscale:{target_id}")
        if self._fail_on == "downscale":
            raise RuntimeError("downscale boom")
        self.weights[target_id] = new_w

    def prune(self, episodic_id):
        self.calls.append(f"prune:{episodic_id}")
        if self._fail_on == "prune":
            raise RuntimeError("prune boom")
        self.episodics.discard(episodic_id)

    def write_gist(self, gist_card, evidence_refs, supersedes):
        self.calls.append("write_gist")
        if self._fail_on == "write_gist":
            raise RuntimeError("gist boom")
        self.cards.append(gist_card)


def _delta(delta_id="d-1"):
    return MemoryCompactionDeltaV1(
        delta_id=delta_id,
        consolidate=[ConsolidateEntryV1(gist_card="a settled theme", evidence_refs=["ol-1"])],
        downscale=[DownscaleEntryV1(target_id="e-1", old_w=0.9, new_w=0.4, reason="settled")],
        prune=[PruneEntryV1(episodic_id="ep-1", salience=0.05, ttl_reason="stale")],
    )


def _approval(delta_id="d-1"):
    """A consent that approves exactly one delta id (stands in for the market's consent gate)."""
    return lambda delta: delta.delta_id == delta_id


# --- gating -------------------------------------------------------------------

def test_disabled_gate_touches_nothing(tmp_path, monkeypatch):
    from app import compaction_applier as ca

    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=False)
    assert r.status == "disabled"
    assert r.applied is False
    assert store.calls == []  # not even a snapshot


def test_not_approved_without_a_consent_gate():
    """No consent gate exists yet (Juniper decision 6, 2026-11-05): with the hot gate on and no
    consent injected, nothing is applied and the store is never touched."""
    from app import compaction_applier as ca

    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), store=store, enabled=True)
    assert r.status == "not_approved"
    assert store.calls == []


def test_not_approved_when_consent_is_for_another_delta():
    from app import compaction_applier as ca

    store = FakeStore()
    r = ca.apply_compaction_delta(_delta("d-1"), _approval("d-OTHER"), store=store, enabled=True)
    assert r.status == "not_approved"
    assert store.calls == []


def test_consent_is_asked_about_this_exact_delta():
    """Approving delta A must not authorize delta B that merely shares a source request."""
    from app import compaction_applier as ca

    delta_a = MemoryCompactionDeltaV1(delta_id="dA", source_request_ids=["r1"],
                                      downscale=[DownscaleEntryV1(target_id="e-1", old_w=0.9, new_w=0.4)])
    delta_b = MemoryCompactionDeltaV1(delta_id="dB", source_request_ids=["r1"],
                                      prune=[PruneEntryV1(episodic_id="ep-1", salience=0.01)])
    asked = []

    def consent(delta):
        asked.append(delta.delta_id)
        return delta.delta_id == "dA"

    assert ca.consent_approves(delta_a, consent) is True
    assert ca.consent_approves(delta_b, consent) is False
    assert asked == ["dA", "dB"]


@pytest.mark.parametrize("answer", [None, 1, "yes", object()])
def test_only_a_literal_true_is_consent(answer):
    from app import compaction_applier as ca

    assert ca.consent_approves(_delta(), lambda d: answer) is False


def test_a_raising_consent_fails_closed():
    from app import compaction_applier as ca

    def boom(delta):
        raise RuntimeError("consent store down")

    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), boom, store=store, enabled=True)
    assert r.status == "not_approved"
    assert store.calls == []


def test_the_retired_policy_frame_is_not_imported():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "compaction_applier.py").read_text(encoding="utf-8")
    assert "orion.schemas.policy_decision_frame" not in src


def test_empty_delta_is_empty(monkeypatch):
    from app import compaction_applier as ca

    r = ca.apply_compaction_delta(
        MemoryCompactionDeltaV1(delta_id="d-empty"),
        _approval("d-empty"),
        store=FakeStore(),
        enabled=True,
    )
    assert r.status == "empty"


# --- happy path + ordering ----------------------------------------------------

def test_snapshot_precedes_every_write(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True, downscale_only=True)
    assert r.status == "applied" and r.applied is True
    # snapshot is the very first call, before any mutation.
    assert store.calls[0] == "snapshot"
    assert "restore" not in store.calls
    # a snapshot artifact was written (§14).
    assert r.snapshot_path is not None
    assert (tmp_path / "d-1" / "snapshot_before.json").exists()


def test_downscale_only_skips_prune(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True, downscale_only=True)
    assert r.downscaled == 1
    assert r.pruned == 0
    assert r.prune_skipped_downscale_only == 1
    assert r.cards_written == 1
    assert not any(c.startswith("prune") for c in store.calls)  # prune never ran
    assert store.weights["e-1"] == 0.4  # downscale applied
    assert store.episodics == {"ep-1", "ep-2"}  # nothing pruned


def test_prune_runs_when_downscale_only_false(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore()
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True, downscale_only=False)
    assert r.pruned == 1
    assert "ep-1" not in store.episodics
    # downscale still ran strictly before prune.
    assert store.calls.index("downscale:e-1") < store.calls.index("prune:ep-1")


# --- rollback -----------------------------------------------------------------

def test_rollback_restores_on_write_error(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore(fail_on="write_gist")
    before_weights = dict(store.weights)
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True, downscale_only=True)
    assert r.status == "rolled_back"
    assert r.applied is False
    assert "restore" in store.calls
    # the downscale that ran before the failure was reverted by the restore.
    assert store.weights == before_weights


def test_snapshot_failure_fails_closed(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    store = FakeStore(fail_snapshot=True)
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True)
    assert r.status == "rolled_back"
    # no mutation was attempted — only the (failed) snapshot.
    assert store.calls == ["snapshot"]


def test_never_raises_even_if_rollback_also_fails(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))

    class DoubleFail(FakeStore):
        def restore(self, snapshot):
            self.calls.append("restore")
            raise RuntimeError("restore boom")

    store = DoubleFail(fail_on="downscale")
    r = ca.apply_compaction_delta(_delta(), _approval(), store=store, enabled=True)
    assert r.status == "rolled_back"
    assert r.error and "rollback_failed" in r.error


def test_module_has_no_default_canonical_store():
    """Phase G must not bind a real canonical-memory store — importing it touches
    nothing. The store is always injected."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "compaction_applier.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    # No sqlalchemy / psycopg import — no path to a live DB from this module.
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    for mod in mods:
        assert "sqlalchemy" not in mod.lower(), "applier must not open a DB itself"
        assert "psycopg" not in mod.lower(), "applier must not open a DB itself"
