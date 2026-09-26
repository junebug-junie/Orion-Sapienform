from copy import deepcopy
import asyncio
import json

import pytest
from pydantic import ValidationError

from app.settings import Settings
from app.worker import handle_help_request, apply_peer_brief_consumed
from orion.curiosity.agency_episode import commit_ask, decision_query, validate_decision
from orion.curiosity.peer_briefs import list_help_requests_from_rows, publish_peer_briefs_consumed, format_soft_nudge
from orion.dev_economics.cursor_limit_events import CursorLimitObservation
from orion.schemas.curiosity_peer import HelpRequestV1, PeerAskExpectationV1, PeerBriefConsumedV1, PeerBriefV1


def request():
    return HelpRequestV1(help_id="help-test",run_id="abcdef012345",mode="world_curiosity",
        question="Which module owns retries?",tried_summary="Need a code pointer",success_criteria="A source-backed answer",
        expectation=PeerAskExpectationV1(expected_reply="Retry logic is in worker.py",if_not_asked="The location stays uncertain",
            alternatives=["hire_peer","inspect_locally"],within_seconds=300))


class CommitGraph:
    def __init__(self):
        self.commits = {}
        self.queries = []

    def graph_query(self, query, params):
        self.queries.append((query, params))
        if "PeerAskCommit" in query:
            return [self.commits.setdefault(params["help_id"], {"commit_token": params["token"], "request_digest": params["digest"]})]
        return []


def settings():
    return Settings(_env_file=None,CURIOSITY_PEER_EPISODES_ENABLED=True)


def clear():
    return CursorLimitObservation(observed=True,state="clear",staleness_sec=1)


def test_commit_ack_precedes_provider_and_restart_does_not_repeat_hire():
    graph = CommitGraph()
    req = request()
    calls = []
    def cursor(h, **kwargs):
        assert h.help_id in graph.commits
        calls.append("called")
        return PeerBriefV1(brief_id="b1",help_id=h.help_id,run_id=h.run_id,peer="cursor_auto",status="ok",summary="worker.py")
    kwargs = dict(settings=settings(),cursor=cursor,observe_limit=clear,persist=lambda b: calls.append("persisted"),begin_episode=lambda h:commit_ask(graph,h))
    assert handle_help_request(req,**kwargs).status == "ok"
    # A fresh parsed request, not an in-process identity/cache, gets no new hire.
    assert handle_help_request(HelpRequestV1.model_validate_json(req.model_dump_json()),**kwargs) is None
    assert calls == ["called","persisted"]


def test_graph_failure_or_missing_forecast_refuses_external_call():
    calls = []
    def unavailable(h):
        raise RuntimeError("graph down")
    with pytest.raises(RuntimeError,match="graph down"):
        handle_help_request(request(),settings=settings(),cursor=lambda *a,**k:calls.append(1),observe_limit=clear,begin_episode=unavailable)
    assert calls == []
    req=request().model_copy(update={"expectation":None})
    with pytest.raises(ValueError,match="explicit expectation"):
        commit_ask(CommitGraph(),req)


def test_changed_request_cannot_reuse_committed_identity():
    graph=CommitGraph()
    req=request()
    assert commit_ask(graph,req)
    with pytest.raises(ValueError,match="different content"):
        commit_ask(graph,req.model_copy(update={"question":"Different question"}))


@pytest.mark.parametrize("kind",["wrong_identity","empty"])
def test_bad_provider_reply_becomes_honest_failure(kind):
    req=request()
    def cursor(h,**kw):
        return PeerBriefV1(brief_id="b",help_id="wrong" if kind=="wrong_identity" else h.help_id,
            run_id=h.run_id,peer="cursor_auto",status="ok",summary="" if kind=="empty" else "answer")
    saved=[]
    result=handle_help_request(req,settings=settings(),cursor=cursor,observe_limit=clear,
        begin_episode=lambda h:True,persist=saved.append)
    assert result.status=="failed"
    assert result.help_id==req.help_id
    assert len(saved)==1


def test_forecast_preserves_original_graph_timestamp():
    req=request()
    row={**req.model_dump(),**req.expectation.model_dump(),"alternatives_json":json.dumps(req.expectation.alternatives),"written_at":1790386258611}
    parsed=list_help_requests_from_rows([row])[0]
    assert int(parsed.written_at.timestamp()*1000)==1790386258611
    assert parsed.expectation==req.expectation
    row["alternatives_json"]="not json"
    assert list_help_requests_from_rows([row])==[]


def decision(**changes):
    row=dict(run_id="abcdef012345",brief_id="b1",help_id="help-test",offered_at=100,completed_at=400,
             disposition="used",reason="The code pointer settled where to look",decision="Inspected the retry loop",
             hop_n=1,written_at=300,hop_written_at=200,hop_note="Read the retry function")
    return {**row,**changes}


def test_used_receipt_needs_completed_run_and_existing_ordered_hop():
    assert validate_decision(decision())["status"]=="attributed_self_report"
    for changes in ({"completed_at":None},{"hop_note":""},{"hop_written_at":99},{"written_at":401},{"reason":""}):
        assert validate_decision(decision(**changes))["status"]=="unverified"
    unused=decision(disposition="not_used",hop_n=None,hop_written_at=None,hop_note=None,decision="Continued independently")
    assert validate_decision(unused)["status"]=="attributed_self_report"
    assert validate_decision(decision(reason="Another explanation"))["evidence_digest"]!=validate_decision(decision())["evidence_digest"]


def test_completion_and_offering_are_distinct_consumer_operations():
    graph=CommitGraph()
    apply_peer_brief_consumed(PeerBriefConsumedV1(brief_ids=["b1"],consumer_run_id="abcdef012345").model_dump(),settings=settings(),graph_client=graph)
    assert "offered_at" in graph.queries[-1][0]
    assert "completed_at" not in graph.queries[-1][0]
    apply_peer_brief_consumed(PeerBriefConsumedV1(consumer_run_id="abcdef012345",phase="completed").model_dump(),settings=settings(),graph_client=graph)
    assert "completed_at" in graph.queries[-1][0]
    assert "coalesce" in graph.queries[-1][0]


def test_producer_publishes_completion_even_without_brief_ids():
    class Bus:
        def __init__(self): self.payloads=[]
        async def publish(self,channel,envelope): self.payloads.append(envelope.payload)
    bus=Bus()
    asyncio.run(publish_peer_briefs_consumed(bus=bus,brief_ids=[],consumer_run_id="abcdef012345",phase="completed"))
    assert bus.payloads[0]["phase"]=="completed"
    assert bus.payloads[0]["consumer_run_id"]=="abcdef012345"
    # Exercise the real Titanium-envelope shape, not just a naked payload.
    graph=CommitGraph()
    apply_peer_brief_consumed(json.dumps({"kind":"curiosity.peer.brief.consumed.v1","payload":bus.payloads[0]}),settings=settings(),graph_client=graph)
    assert "completed_at" in graph.queries[-1][0]
    with pytest.raises(ValidationError):
        PeerBriefConsumedV1(phase="completed")


def test_prompt_has_forecast_comparison_and_no_invented_step_instruction():
    req=request()
    brief=PeerBriefV1(brief_id="b1",help_id=req.help_id,run_id=req.run_id,peer="cursor_auto",status="ok",summary="worker.py")
    prompt="\n".join(format_soft_nudge([brief],consumer_run_id="abcdef012345"))
    assert "PeerAskCommit" in prompt and "PeerBriefDecision" in prompt
    assert "do not invent a step" in prompt
    assert "not proof that it caused" in prompt
    assert "PeerBriefOffer" in decision_query("abcdef012345")
    with pytest.raises(ValueError): decision_query("' DELETE n")


def test_refused_brief_id_is_visible_for_not_used_receipt():
    req=request()
    brief=PeerBriefV1(brief_id="b-refused",help_id=req.help_id,run_id=req.run_id,peer="cursor_auto",status="refused_budget")
    prompt="\n".join(format_soft_nudge([brief],consumer_run_id="abcdef012345"))
    assert "b-refused" in prompt and "not_used" in prompt
