from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from orion.schemas.harness_finalize import HarnessPostTurnClosureV1

from app.turn_referent_store import persist_turn_referent


def test_persist_turn_referent_upserts_on_unresolved_closure() -> None:
    conn = MagicMock()
    begin_ctx = MagicMock()
    begin_ctx.__enter__ = MagicMock(return_value=conn)
    begin_ctx.__exit__ = MagicMock(return_value=False)
    engine = MagicMock()
    engine.begin.return_value = begin_ctx

    closure = HarnessPostTurnClosureV1(
        correlation_id="corr-1",
        outcome_molecule_id="out-1",
        verdict_molecule_id="ver-1",
        surprise_unresolved=True,
        user_message_excerpt="You asked about the deploy.",
        stance_imperative="Name the risk.",
        thought_event_id="te-1",
    )
    ok = persist_turn_referent(closure, engine=engine, now=datetime(2026, 7, 7, tzinfo=timezone.utc))
    assert ok is True
    conn.execute.assert_called_once()
    params = conn.execute.call_args.args[1]
    assert params["correlation_id"] == "corr-1"
    assert params["coalition_ref"] == "harness_closure:corr-1"
    assert params["user_message_excerpt"] == "You asked about the deploy."


def test_persist_turn_referent_skips_when_surprise_resolved() -> None:
    engine = MagicMock()
    closure = HarnessPostTurnClosureV1(
        correlation_id="corr-2",
        outcome_molecule_id="out-2",
        verdict_molecule_id="ver-2",
        surprise_unresolved=False,
        user_message_excerpt="Some message.",
        stance_imperative="Some stance.",
    )
    ok = persist_turn_referent(closure, engine=engine)
    assert ok is False
    engine.begin.assert_not_called()


def test_persist_turn_referent_skips_when_excerpts_empty() -> None:
    engine = MagicMock()
    closure = HarnessPostTurnClosureV1(
        correlation_id="corr-3",
        outcome_molecule_id="out-3",
        verdict_molecule_id="ver-3",
        surprise_unresolved=True,
        user_message_excerpt="",
        stance_imperative="   ",
    )
    ok = persist_turn_referent(closure, engine=engine)
    assert ok is False
    engine.begin.assert_not_called()


def test_persist_turn_referent_fail_open_on_db_error() -> None:
    engine = MagicMock()
    engine.begin.side_effect = RuntimeError("db down")
    closure = HarnessPostTurnClosureV1(
        correlation_id="corr-4",
        outcome_molecule_id="out-4",
        verdict_molecule_id="ver-4",
        surprise_unresolved=True,
        user_message_excerpt="Deploy risk?",
        stance_imperative="Name it.",
    )
    ok = persist_turn_referent(closure, engine=engine)
    assert ok is False
