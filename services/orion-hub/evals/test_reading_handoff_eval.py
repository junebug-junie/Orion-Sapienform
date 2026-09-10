"""Offline behavioral fixtures: useful source candidates, lineage, empty/unsafe output.

Recorded model-shaped responses are injected; this does not measure live model quality.
"""
import asyncio
import json
from uuid import uuid4

import pytest

from orion.schemas.reading import ReadingRequestedV1
from orion.schemas.world_pulse_read import WorldPulseReadSeedV1
from orion.substrate.adapters.world_pulse_read import map_world_pulse_read_handoff_to_substrate
from scripts.world_pulse_read_pipeline import GenerateOutcome, WorldPulseReadPipeline, _build_stage1_prompt


@pytest.mark.parametrize("context,requester", [("unified_chat", "juniper"), ("curiosity", "orion"), ("world_pulse", "world_pulse")])
def test_attributed_learning_survives_all_ingresses(context, requester):
    req = ReadingRequestedV1(url="https://example.org/study", requested_by=requester, invocation_context=context,
                             why_now="Compare this result with the previous study", parent_run_id="parent-run", parent_trace_id="parent-trace")
    seed = WorldPulseReadSeedV1(seed_id="reading:" + str(req.request_id), kind="reading", run_id="parent-run",
                               url=str(req.url), request=req)
    pipe = object.__new__(WorldPulseReadPipeline)
    async def generate(prompt, trace):
        assert "world-pulse article" not in prompt
        assert req.why_now in prompt
        return GenerateOutcome(json.dumps({
            "what_i_learned": "This source reports improved recall in its study; independent replication is still needed.",
            "candidate_priors": [{"claim": "The reported intervention may improve recall under the tested conditions.", "confidence": 0.4}],
            "concept_candidates": [{"label": "retrieval practice", "definition": "Recall practice compared with repeated exposure."}],
            "open_threads": ["Does this generalize to long-term recall?"],
        }))
    pipe._generate = generate
    handoff = asyncio.run(pipe._stage1_read(seed))
    record = map_world_pulse_read_handoff_to_substrate(handoff)
    assert handoff.seed_ref.request == req
    assert record.nodes and not record.edges
    node = record.nodes[0]
    assert node.metadata["reading_request"]["requested_by"] == requester
    assert str(req.url) in node.provenance.evidence_refs
    assert str(req.request_id) in node.provenance.evidence_refs


@pytest.mark.parametrize("response", [
    {"what_i_learned": "   "},
    {"what_i_learned": "A claim", "cypher": "CREATE (n)"},
    {"what_i_learned": "A claim", "rdf": "trusted triple"},
])
def test_empty_or_direct_graph_output_is_rejected(response):
    seed = WorldPulseReadSeedV1(seed_id="s", kind="finding", run_id="r", url="https://example.org/source")
    pipe = object.__new__(WorldPulseReadPipeline)
    async def generate(prompt, trace):
        return GenerateOutcome(json.dumps(response))
    pipe._generate = generate
    with pytest.raises(ValueError):
        asyncio.run(pipe._stage1_read(seed))
