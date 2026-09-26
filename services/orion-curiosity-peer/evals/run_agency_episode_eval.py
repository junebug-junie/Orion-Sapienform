#!/usr/bin/env python3
"""Exercise the ask lifecycle on a disposable FalkorDB, never production.

Starts/stops its own container with an ephemeral loopback port and no mounted
volumes. Provider and bus delivery are deterministic in-process fixtures;
commitment, matching, offer, completion and decision reads use the real engine.
No Cursor/Claude invocation or production bus connection is possible here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

for path in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[3]):
    sys.path.insert(0,str(path))

from app.settings import Settings
from app.worker import handle_help_request, apply_peer_brief_consumed
from orion.curiosity.agency_episode import commit_ask, decision_query, validate_decision
from orion.curiosity.peer_brief_persist import persist_peer_brief
from orion.curiosity.peer_briefs import publish_peer_briefs_consumed
from orion.dev_economics.cursor_limit_events import CursorLimitObservation
from orion.graph.falkor_client import RedisGraphQueryClient
from orion.schemas.curiosity_peer import HelpRequestV1, PeerAskExpectationV1, PeerBriefV1


def docker(*args):
    return subprocess.check_output(["docker",*args],text=True).strip()


def exercise(graph):
    req=HelpRequestV1(help_id="eval-ask",run_id="abcdef012345",mode="world_curiosity",
        question="Where does retry live?",tried_summary="Need a source pointer",success_criteria="A source-backed location",
        expectation=PeerAskExpectationV1(expected_reply="worker.py owns retry",if_not_asked="Location remains unknown",
            alternatives=["hire_peer","inspect_locally"],within_seconds=300))
    graph.graph_query("CREATE (:HelpRequest {help_id:$help_id,run_id:$run_id,written_at:timestamp()})",
                      {"help_id":req.help_id,"run_id":req.run_id})
    calls=[]
    def provider(h,**kwargs):
        rows=graph.graph_query("MATCH (c:PeerAskCommit {help_id:$help_id}) RETURN c.committed_at AS committed_at",{"help_id":h.help_id})
        assert len(rows)==1 and rows[0]["committed_at"]>0
        calls.append(h.help_id)
        return PeerBriefV1(brief_id="eval-brief",help_id=h.help_id,run_id=h.run_id,peer="cursor_auto",status="ok",summary="retry.py owns it, not worker.py",evidence_pointers=["retry.py"])
    persisted=[]
    def persist(brief):
        persist_peer_brief(brief=brief,graph_execute=graph.graph_query,bus_publish=lambda channel,env:persisted.append(env))
    settings=Settings(_env_file=None,CURIOSITY_PEER_EPISODES_ENABLED=True)
    kwargs=dict(settings=settings,cursor=provider,observe_limit=lambda:CursorLimitObservation(observed=True,state="clear",staleness_sec=1),
                begin_episode=lambda h:commit_ask(graph,h),persist=persist)
    brief=handle_help_request(req,**kwargs)
    assert brief.status=="ok" and len(persisted)==1
    assert handle_help_request(HelpRequestV1.model_validate_json(req.model_dump_json()),**kwargs) is None
    assert calls==[req.help_id]
    returned=graph.graph_query("MATCH (c:PeerAskCommit)-[:RETURNED]->(b:PeerBrief) RETURN c.committed_at AS committed_at,c.responded_at AS responded_at")
    assert len(returned)==1 and returned[0]["responded_at"]>=returned[0]["committed_at"]

    class Bus:
        def __init__(self): self.messages=[]
        async def publish(self,channel,envelope): self.messages.append(envelope.model_dump_json())
    bus=Bus()
    consumer_run="abcdef654321"
    asyncio.run(publish_peer_briefs_consumed(bus=bus,brief_ids=[brief.brief_id],consumer_run_id=consumer_run))
    apply_peer_brief_consumed(bus.messages[-1],settings=settings,graph_client=graph)
    # A replayed brief must not reset its consumed bit or original timestamp.
    before=graph.graph_query("MATCH (b:PeerBrief) RETURN b.written_at AS written_at,b.consumed AS consumed")
    persist(brief)
    assert graph.graph_query("MATCH (b:PeerBrief) RETURN b.written_at AS written_at,b.consumed AS consumed")==before

    graph.graph_query("CREATE (:Hop {run_id:$run_id,n:1,note:'Inspected retry.py using peer pointer',written_at:timestamp()}) "
        "CREATE (:PeerBriefDecision {run_id:$run_id,brief_id:$brief_id,disposition:'used', "
        "decision:'Inspect retry.py instead of worker.py',reason:'The peer contradicted the forecast with a source pointer',hop_n:1,written_at:timestamp()})",
        {"run_id":consumer_run,"brief_id":brief.brief_id})
    rows=graph.graph_query(decision_query(consumer_run))
    assert len(rows)==1 and validate_decision(rows[0])["status"]=="unverified"
    asyncio.run(publish_peer_briefs_consumed(bus=bus,brief_ids=[],consumer_run_id=consumer_run,phase="completed"))
    apply_peer_brief_consumed(bus.messages[-1],settings=settings,graph_client=graph)
    rows=graph.graph_query(decision_query(consumer_run))
    assert len(rows)==1 and validate_decision(rows[0])["status"]=="attributed_self_report"
    # Duplicate completion does not rewrite its acknowledgment time.
    apply_peer_brief_consumed(bus.messages[-1],settings=settings,graph_client=graph)
    assert graph.graph_query(decision_query(consumer_run))==rows
    # Another run cannot claim to have used a brief it was never offered.
    assert graph.graph_query(decision_query("abcdef999999"))==[]
    return {"precall_graph_ack":True,"duplicate_hire_blocked":True,"response_matched":True,
            "brief_replay_preserves_consumption":True,"offered_not_completed":True,
            "completed_decision_joins_hop":True,"completion_replay_stable":True,"wrong_run_rejected":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",default="falkordb/falkordb:latest")
    args=parser.parse_args()
    name="orion-agency-eval-"+uuid.uuid4().hex[:10]
    docker("run","--rm","-d","--name",name,"-p","127.0.0.1::6379",args.image)
    try:
        port=int(docker("port",name,"6379/tcp").rsplit(":",1)[1])
        import redis
        client=redis.Redis(host="127.0.0.1",port=port,socket_timeout=1)
        for _ in range(100):
            try:
                if client.ping(): break
            except redis.RedisError:
                time.sleep(0.1)
        else:
            raise RuntimeError("isolated graph did not start")
        graph=RedisGraphQueryClient(uri=f"redis://127.0.0.1:{port}/0",graph_name="agency_eval")
        checks=exercise(graph)
        print(json.dumps({"eval":"isolated_agency_ask_lifecycle","checks":checks,"passed":len(checks)},indent=2))
    finally:
        docker("stop",name)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
