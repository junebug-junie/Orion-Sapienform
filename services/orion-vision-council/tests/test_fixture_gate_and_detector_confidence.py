"""#3387: the council gate ignores label changes made only of stationary
fixtures; confidence comes from detector scores, never from the prompt
example; each model-written row names the model that served it.

Live 2026-11-15: 925 walkway rows a day, 9,165 of 9,242 with confidence 0.8
(the literal that was in the prompt example), no model name on any row.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.vision import VisionWindowPayload  # noqa: E402

from app import main as council_main  # noqa: E402
from app.evidence_grounding import build_person_presence_fallback  # noqa: E402
from app.evidence_transition import EvidenceTransitionTracker, snapshot_from_window  # noqa: E402
from app.interpretation import (  # noqa: E402
    apply_detector_confidence,
    build_interpretation_prompt,
    detector_confidence,
    parse_llm_content,
    project_interpretation_to_events,
)
from app.main import CouncilService  # noqa: E402


def _window(
    *,
    hard_labels: list[str] | None = None,
    believed: list[str] | None = None,
    fixture_only: list[str] | None = None,
    label_scores: dict[str, float] | None = None,
    stream_id: str = "walkway",
) -> VisionWindowPayload:
    evidence: dict = {"hard_labels": hard_labels or [], "soft_labels": [], "host_person_hits": 0, "caption_count": 0}
    if believed is not None:
        evidence["believed_hard_labels"] = believed
        evidence["belief"] = {"schema": "scene_belief.v1"}
    if fixture_only is not None:
        evidence["fixture_only_labels"] = fixture_only
    if label_scores is not None:
        evidence["label_scores"] = label_scores
    return VisionWindowPayload(
        window_id="w1", start_ts=1.0, end_ts=2.0, stream_id=stream_id,
        summary={"object_counts": {}, "top_labels": [], "captions": [], "item_count": 1,
                 "detection_count": 0, "evidence": evidence},
        artifact_ids=["art-1"],
    )


def _evaluate(tracker: EvidenceTransitionTracker, window: VisionWindowPayload, now: float, ttl: float = 600.0):
    return tracker.evaluate(stream_key="walkway", snapshot=snapshot_from_window(window), now=now, max_refresh_sec=ttl)


# --- (a) the gate ----------------------------------------------------------------------------


def test_parked_car_lost_and_refound_does_not_trigger() -> None:
    """The night flood: believed {vehicle} <-> {} while the only vehicle is a
    parked car. Neither direction is a change in the scene."""
    tracker = EvidenceTransitionTracker()
    seen = _window(hard_labels=["vehicle"], believed=["vehicle"], fixture_only=["vehicle"])
    tracker.record_interpretation(stream_key="walkway", snapshot=snapshot_from_window(seen), now=100.0)

    lost = _evaluate(tracker, _window(hard_labels=[], believed=[], fixture_only=["vehicle"]), 110.0)
    assert lost.interpret is False and lost.reason == "fixture_only_change"

    refound = _evaluate(tracker, seen, 120.0)
    assert refound.interpret is False and refound.reason == "stable_scene"

    # And from the other side: last interpreted with nothing in view.
    tracker.record_interpretation(
        stream_key="walkway", snapshot=snapshot_from_window(_window(hard_labels=[], believed=[])), now=200.0)
    back = _evaluate(tracker, seen, 210.0)
    assert back.interpret is False and back.reason == "fixture_only_change"


def test_a_moving_vehicle_still_triggers() -> None:
    """Same label flip, but the vehicle in view is not a fixture: a real arrival."""
    tracker = EvidenceTransitionTracker()
    tracker.record_interpretation(
        stream_key="walkway", snapshot=snapshot_from_window(_window(hard_labels=[], believed=[])), now=100.0)
    arrived = _evaluate(tracker, _window(hard_labels=["vehicle"], believed=["vehicle"], fixture_only=[]), 110.0)
    assert arrived.interpret is True and arrived.reason == "salient_labels_changed"


def test_a_person_beside_a_parked_car_still_triggers() -> None:
    tracker = EvidenceTransitionTracker()
    parked = _window(hard_labels=["vehicle"], believed=["vehicle"], fixture_only=["vehicle"])
    tracker.record_interpretation(stream_key="walkway", snapshot=snapshot_from_window(parked), now=100.0)
    person = _evaluate(
        tracker, _window(hard_labels=["person", "vehicle"], believed=["person", "vehicle"], fixture_only=["vehicle"]),
        110.0)
    assert person.interpret is True and person.reason == "person_entered"


def test_real_change_is_seen_after_an_ignored_fixture_flip() -> None:
    """The ignored flip must not overwrite what was last interpreted: a dog
    arriving while the parked car is out of sight is still a change."""
    tracker = EvidenceTransitionTracker()
    parked = _window(hard_labels=["vehicle"], believed=["vehicle"], fixture_only=["vehicle"])
    tracker.record_interpretation(stream_key="walkway", snapshot=snapshot_from_window(parked), now=100.0)
    assert _evaluate(tracker, _window(believed=[], fixture_only=["vehicle"]), 110.0).interpret is False
    dog = _evaluate(tracker, _window(hard_labels=["dog"], believed=["dog"], fixture_only=["vehicle"]), 120.0)
    assert dog.interpret is True and dog.reason == "salient_labels_changed"


def test_fixture_only_change_still_refreshes_on_the_ttl() -> None:
    tracker = EvidenceTransitionTracker()
    parked = _window(hard_labels=["vehicle"], believed=["vehicle"], fixture_only=["vehicle"])
    tracker.record_interpretation(stream_key="walkway", snapshot=snapshot_from_window(parked), now=100.0)
    late = _evaluate(tracker, _window(believed=[], fixture_only=["vehicle"]), 100.0 + 600.0)
    assert late.interpret is True and late.reason == "refresh_ttl"


def test_window_without_fixture_evidence_keeps_the_old_rule() -> None:
    """Room cameras (and an older orion-vision-window) send no fixture list."""
    tracker = EvidenceTransitionTracker()
    tracker.record_interpretation(
        stream_key="walkway", snapshot=snapshot_from_window(_window(hard_labels=["vehicle"])), now=100.0)
    gone = _evaluate(tracker, _window(hard_labels=[]), 110.0)
    assert gone.interpret is True and gone.reason == "salient_labels_changed"


# --- (b) confidence --------------------------------------------------------------------------


def test_prompt_example_carries_no_confidence_number() -> None:
    prompt = build_interpretation_prompt(_window(hard_labels=["vehicle"], label_scores={"vehicle": 0.31}))
    shape = prompt.split("Required output shape", 1)[1].split("Rules:", 1)[0]
    assert '"confidence"' not in shape
    assert "0.8" not in shape
    assert "Do not output a confidence value" in prompt


def test_confidence_is_the_detector_score_of_the_named_label() -> None:
    w = _window(hard_labels=["person", "vehicle"], label_scores={"vehicle": 0.2712, "person": 0.55})
    assert detector_confidence(w, ["vehicle"]) == 0.2712
    assert detector_confidence(w, ["Vehicle"]) == 0.2712           # label case does not matter
    assert detector_confidence(w, ["person", "vehicle"]) == 0.2712  # the weakest named label
    assert detector_confidence(w, []) == 0.2712                     # no label named: the whole window
    assert detector_confidence(w, ["vehicle", "street"]) == 0.2712  # an unseen name cannot lower a measured score


def test_no_detector_evidence_is_none_not_a_number() -> None:
    assert detector_confidence(_window(hard_labels=[]), ["vehicle"]) is None             # nothing detected
    assert detector_confidence(_window(hard_labels=[], label_scores={}), []) is None
    w = _window(hard_labels=["person"], label_scores={"person": 0.6})
    assert detector_confidence(w, ["vehicle"]) is None    # the row names something the detector did not see
    assert detector_confidence(None, ["vehicle"]) is None
    # Garbage scores are not evidence.
    assert detector_confidence(_window(label_scores={"vehicle": "high", "dog": True}), ["vehicle"]) is None


def _llm_json(confidence: float | None = 0.8, entities=("vehicle",)) -> str:
    event = {"event_type": "visual_observation", "narrative": "A vehicle is observed in the walkway.",
             "entities": list(entities), "tags": [], "salience": 0.7, "evidence_refs": ["art-1"]}
    if confidence is not None:
        event["confidence"] = confidence
    return json.dumps({"schema_version": "1.0", "window_id": "w1", "scene_summary": "A vehicle.",
                       "salient_observations": [], "uncertainties": [], "event_candidates": [event],
                       "evidence_refs": []})


def test_the_models_own_confidence_is_replaced() -> None:
    """The model still echoes 0.8; the row gets the detector's number."""
    w = _window(hard_labels=["vehicle"], label_scores={"vehicle": 0.2931})
    interp = parse_llm_content(_llm_json(0.8), w).interpretation
    assert interp is not None and interp.event_candidates[0].confidence == 0.8
    measured = apply_detector_confidence(interp, w)
    assert measured.event_candidates[0].confidence == 0.2931


def test_confidence_varies_with_the_detector_and_reads_low() -> None:
    values = set()
    for score in (0.2501, 0.33, 0.47, 0.64):
        w = _window(hard_labels=["vehicle"], label_scores={"vehicle": score})
        interp = apply_detector_confidence(parse_llm_content(_llm_json(0.8), w).interpretation, w)
        values.add(interp.event_candidates[0].confidence)
    assert values == {0.2501, 0.33, 0.47, 0.64}


def test_person_fallback_uses_the_person_score_not_085() -> None:
    w = _window(hard_labels=["person"], label_scores={"person": 0.4217})
    assert build_person_presence_fallback(w).event_candidates[0].confidence == 0.4217
    assert build_person_presence_fallback(_window(hard_labels=[])).event_candidates[0].confidence is None


# --- (c) serving model -----------------------------------------------------------------------


def test_model_written_rows_name_the_serving_model() -> None:
    w = _window(hard_labels=["vehicle"], label_scores={"vehicle": 0.3})
    interp = parse_llm_content(_llm_json(None), w).interpretation
    payload = project_interpretation_to_events(interp, w, model_used="/models/gguf/Qwen3.5-9B-UD-Q5_K_XL.gguf")
    assert payload.events[0].evidence_refs == ["art-1", "model:/models/gguf/Qwen3.5-9B-UD-Q5_K_XL.gguf"]
    # No model known: nothing invented.
    assert project_interpretation_to_events(interp, w).events[0].evidence_refs == ["art-1"]


def test_host_built_rows_do_not_claim_a_model() -> None:
    w = _window(hard_labels=["person"], label_scores={"person": 0.5})
    payload = project_interpretation_to_events(build_person_presence_fallback(w), w, model_used="qwen")
    assert payload.events[0].evidence_refs == ["art-1"]


def _env() -> BaseEnvelope:
    return BaseEnvelope(kind="vision.window", source=ServiceRef(name="test", version="0"), payload={})


def test_process_window_publishes_measured_confidence_and_the_served_model(monkeypatch) -> None:
    """Whole path: gateway reply -> parse -> finalize -> published bundle."""
    monkeypatch.setattr(council_main.settings, "COUNCIL_TRANSITION_GATE_ENABLED", False)
    monkeypatch.setattr(council_main.settings, "COUNCIL_UNRESOLVED_ENABLED", False)
    svc = CouncilService()
    published = []

    async def _publish(channel, env):
        published.append(env)

    async def _llm(prompt, source_env):
        # What _call_llm_raw does on a real reply.
        council_main._SERVED_MODEL.set(council_main._model_used_from_reply(
            {"content": "x", "model_used": "/models/gguf/Qwen3.5-9B-UD-Q5_K_XL.gguf"}))
        return _llm_json(0.8)

    monkeypatch.setattr(svc.bus, "publish", _publish)
    monkeypatch.setattr(svc, "_call_llm_raw", _llm)
    w = _window(hard_labels=["vehicle"], label_scores={"vehicle": 0.2764})
    env = _env().model_copy(update={"payload": w.model_dump(mode="json")})
    asyncio.run(svc._process_window(env))

    assert len(published) == 1
    payload = published[0].payload
    events = payload["events"] if isinstance(payload, dict) else payload.events
    event = events[0] if isinstance(events[0], dict) else events[0].model_dump()
    assert event["confidence"] == 0.2764
    assert event["evidence_refs"][-1] == "model:/models/gguf/Qwen3.5-9B-UD-Q5_K_XL.gguf"


def test_model_used_is_read_from_the_gateway_reply() -> None:
    f = council_main._model_used_from_reply
    assert f({"model_used": " qwen-9b "}) == "qwen-9b"
    assert f({"raw": {"model": "from-raw"}}) == "from-raw"
    assert f({"content": "x"}) is None and f(None) is None and f({"model_used": ""}) is None


def test_gate_skip_end_to_end_makes_no_llm_call() -> None:
    svc = CouncilService()
    parked = _window(hard_labels=["vehicle"], believed=["vehicle"], fixture_only=["vehicle"])
    svc._evidence_transition.record_interpretation(stream_key="walkway", snapshot=snapshot_from_window(parked))
    lost = _window(hard_labels=[], believed=[], fixture_only=["vehicle"])
    with patch.object(svc, "_call_llm_raw", new_callable=AsyncMock) as mock_llm:
        interp, outcome = asyncio.run(svc._generate_interpretation(lost, _env()))
    assert interp is None and outcome.parse_mode == "stable_scene"
    assert outcome.salvage_warnings == ["fixture_only_change"]
    mock_llm.assert_not_awaited()


def test_the_real_llm_call_reads_the_served_model_from_the_gateway_reply(monkeypatch) -> None:
    """No stub of _call_llm_raw: the gateway reply envelope goes through the
    real call, and the model it names ends up on the published row."""
    import uuid

    from orion.core.bus.codec import OrionCodec

    monkeypatch.setattr(council_main.settings, "COUNCIL_TRANSITION_GATE_ENABLED", False)
    monkeypatch.setattr(council_main.settings, "COUNCIL_UNRESOLVED_ENABLED", False)

    class FakeRpcBus:
        codec = OrionCodec()
        replies = [
            {"content": _llm_json(0.8), "model_used": "/models/gguf/Qwen3.5-9B-UD-Q5_K_XL.gguf", "raw": {}},
            # The next call fails (empty reply). Its window must not inherit the first model name.
            {"content": "", "raw": {}},
        ]

        async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
            out = BaseEnvelope(kind="llm.chat.result", source=ServiceRef(name="gw", version="0"),
                               correlation_id=uuid.uuid4(), payload=self.replies.pop(0))
            return {"data": self.codec.encode(out)}

    svc = CouncilService()
    svc._rpc_bus = FakeRpcBus()
    published = []

    async def _publish(channel, env):
        published.append(env)

    monkeypatch.setattr(svc.bus, "publish", _publish)
    w = _window(hard_labels=["vehicle"], label_scores={"vehicle": 0.31})
    env = _env().model_copy(update={"payload": w.model_dump(mode="json")})

    async def _both():
        await svc._process_window(env)
        # Same task, so the same context: a leak would show here.
        return await svc._generate_interpretation(w, _env(), allow_transition_gate=False)

    interp, outcome = asyncio.run(_both())
    payload = published[0].payload
    events = payload["events"] if isinstance(payload, dict) else payload.events
    event = events[0] if isinstance(events[0], dict) else events[0].model_dump()
    assert event["evidence_refs"][-1] == "model:/models/gguf/Qwen3.5-9B-UD-Q5_K_XL.gguf"
    assert event["confidence"] == 0.31
    assert interp is None and outcome.model_used is None
    assert council_main._SERVED_MODEL.get() is None


def test_activity_guess_is_tagged_and_capped_whatever_the_score() -> None:
    """Review finding: the caption_inferred tag must not depend on the number."""
    from app.evidence_grounding import enforce_evidence_grounding
    from orion.schemas.vision import VisionEventCandidateV1, VisionSceneInterpretationV1

    def run(conf):
        w = _window(hard_labels=["person"], label_scores={"person": 0.9}, stream_id="cam0")
        w = w.model_copy(update={"summary": {**w.summary, "captions": ["a person reading"]}})
        interp = VisionSceneInterpretationV1(window_id="w1", scene_summary="s", event_candidates=[
            VisionEventCandidateV1(event_type="human_activity", narrative="A person is reading.",
                                   entities=["person"], confidence=conf)])
        out, notes = enforce_evidence_grounding(interp, w)
        cand = out.event_candidates[0]
        return cand.confidence, cand.tags, notes

    assert run(0.62) == (0.4, ["caption_inferred"], ["capped_confidence:human_activity"])
    assert run(0.31) == (0.31, ["caption_inferred"], ["capped_confidence:human_activity"])   # never raised
    assert run(None) == (None, ["caption_inferred"], ["capped_confidence:human_activity"])   # never invented
