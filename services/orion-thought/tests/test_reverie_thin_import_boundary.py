"""Import-boundary tests: orion-thought must not load orion.substrate."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.reverie.referent_loader import TurnReferentRow
from orion.reverie.semantic_lift import resolve_concern_cards
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import ConcernCardV1


def test_importing_reverie_does_not_load_substrate_or_requests() -> None:
    """The guardrail the reverie function-local imports exist to protect.

    Importing `app.reverie` must not drag in `orion.substrate` (graph engine)
    or `requests` at module scope. Run in a fresh interpreter so prior test
    imports can't mask a regression.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    service_root = Path(__file__).resolve().parents[1]  # services/orion-thought
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(service_root), str(repo_root)])
    code = (
        "import app.reverie;"
        "import sys;"
        "sub=[m for m in sys.modules if m.startswith('orion.substrate')];"
        "assert not sub, sub;"
        "assert 'requests' not in sys.modules;"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(service_root),
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_calling_reverie_tick_helpers_does_not_load_substrate() -> None:
    """Real teeth: the tick path (`derive_salience`, `build_salience_trace`) must
    not import `orion.substrate` at CALL time either.

    The prior module-scope-only check passed in a venv that has `requests`, but in
    the thin `orion-thought` container (no `requests`) a call-time
    `from orion.substrate.attention...` crashed every reverie tick. This exercises
    both the v2-on and v2-off branches and asserts `orion.substrate` never loads.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    service_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(service_root), str(repo_root)])
    env["ORION_ATTENTION_SALIENCE_V2_ENABLED"] = "true"
    code = (
        "import sys\n"
        "from datetime import datetime, timezone\n"
        "from orion.schemas.attention_frame import ("
        " AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1)\n"
        "import app.reverie as r\n"
        "loop = OpenLoopV1(id='ol-1', description='x', salience=0.7,"
        " salience_features={'evidence_strength': 0.8})\n"
        "b = AttentionBroadcastProjectionV1("
        " frame=AttentionFrameV1(open_loops=[loop]),"
        " selected_open_loop_id='ol-1', coalition_stability_score=0.4)\n"
        "assert r.derive_salience(b) > 0\n"
        "t = r.build_salience_trace(b, correlation_id='c1')\n"
        "assert t is not None and t.weights_version == 'seed-v1'\n"
        "sub=[m for m in sys.modules if m.startswith('orion.substrate')]\n"
        "assert not sub, sub\n"
        "assert 'requests' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(service_root),
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_stable_hash_id_deterministic() -> None:
    from orion.core.ids import stable_hash_id

    a = stable_hash_id("concern", ["harness_closure:abc"])
    b = stable_hash_id("concern", ["harness_closure:abc"])
    assert a == b
    assert a == "concern_0ebb4258d3d772159cf641a6"


def test_concern_card_from_harness_turn_without_substrate_import() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-thin",
        user_message_excerpt="Will the deploy slip if we cut testing?",
        stance_imperative="Name the testing tradeoff before reassuring.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    assert card.card_id.startswith("concern_")


def test_resolve_concern_cards_builds_real_cards_unmocked() -> None:
    row = TurnReferentRow(
        correlation_id="corr-thin",
        coalition_ref="harness_closure:corr-thin",
        user_message_excerpt="What's my last PR title?",
        stance_imperative="Search PR metadata for the most recent pull request title.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    loader = MagicMock()
    loader.load_by_coalition_ref.return_value = row
    broadcast = AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(
            open_loops=[
                OpenLoopV1(
                    id="ol-1",
                    description="pr title",
                    source_refs=["harness_closure:corr-thin"],
                )
            ]
        ),
        attended_node_ids=["harness_closure:corr-thin"],
        selected_open_loop_id="ol-1",
        coalition_stability_score=0.5,
    )
    cards = resolve_concern_cards(broadcast, referent_loader=loader)
    assert len(cards) == 1
    assert "PR title" in cards[0].human_text
