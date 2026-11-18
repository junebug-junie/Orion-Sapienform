"""node:prediction.room_joint is a WORLD source (#3575; Juniper's #3580 decision), the body joint is not.

The room joint (office air and light, family A) can author a goal through the attended-world rule, is
excluded from body-only gates, and stays current in world-first for two of its 5-min writes (600 s),
not perception's 180 s: written once per scored bucket, it would otherwise read absent 2 min of every 5.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.attention.world_first import (
    EXTERNAL_MAX_AGE_SEC_BY_NODE,
    PE_STALENESS_HORIZON_SEC,
    PERCEPTION_MAX_AGE_SEC,
    node_candidate,
    source_kind_for_node,
)
from orion.schemas.attention_candidate import ROOM_JOINT_NODE_ID, is_world_source_id
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
from orion.substrate.prediction_error_freshness import is_retired_prediction_error_node
from orion.substrate.prediction_joint_surprise import BODY_JOINT_NODE_ID
from orion.substrate.prediction_joint_surprise import ROOM_JOINT_NODE_ID as FOLD_ROOM_ID

NOW = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)


def _cand(node_id: str, age: float):
    m = PredictionErrorMagnitudeV1(value=0.6, age_sec=age, percentile_now=0.995, n_readings_7d=2000,
                                   band="unusual", trend="flat")
    return node_candidate(node_id=node_id, label=node_id, magnitude=m, observed_at=NOW, now=NOW)


def test_room_is_world_body_is_body_and_the_ids_agree():
    assert FOLD_ROOM_ID == ROOM_JOINT_NODE_ID
    assert is_world_source_id(ROOM_JOINT_NODE_ID) and source_kind_for_node(ROOM_JOINT_NODE_ID) == "external"
    assert not is_world_source_id(BODY_JOINT_NODE_ID) and source_kind_for_node(BODY_JOINT_NODE_ID) == "internal"


@pytest.mark.parametrize("age,absent", [(240.0, False), (599.0, False), (601.0, True)])
def test_room_joint_stays_current_for_two_of_its_writes(age, absent):
    c = _cand(ROOM_JOINT_NODE_ID, age)
    assert (c.absent_reason is not None) is absent


def test_perception_keeps_its_180_s():
    assert _cand("node:substrate.perception", PERCEPTION_MAX_AGE_SEC + 1).absent_reason is not None


def test_per_node_max_age_never_exceeds_the_pe_staleness_horizon():
    assert all(v <= PE_STALENESS_HORIZON_SEC for v in EXTERNAL_MAX_AGE_SEC_BY_NODE.values())


def test_successors_are_live_and_the_predecessor_is_retired():
    assert is_retired_prediction_error_node("node:substrate.biometrics")
    assert not is_retired_prediction_error_node(BODY_JOINT_NODE_ID)
    assert not is_retired_prediction_error_node(ROOM_JOINT_NODE_ID)
