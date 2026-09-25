"""Offline behavioral fixtures: useful source candidates, lineage, empty/unsafe output.

Recorded model-shaped responses are injected; this does not measure live model quality.
"""
import asyncio
import json
from uuid import uuid4

import pytest

from orion.harness.finalize import canonicalize_structured_output
from orion.schemas.reading import ReadingRequestedV1, SourceFetchEvidenceV1
from orion.schemas.world_pulse_read import WorldPulseReadSeedV1
from orion.substrate.adapters.world_pulse_read import map_world_pulse_read_handoff_to_substrate
from scripts.world_pulse_read_pipeline import (
    GenerateOutcome,
    NoReadEvidenceError,
    WorldPulseReadPipeline,
    _build_stage1_prompt,
)


def _fetched(url: str) -> list[SourceFetchEvidenceV1]:
    """What the governor reports when the turn really fetched the source."""
    return [SourceFetchEvidenceV1(url=url, tool_name="WebFetch", content_chars=1800)]


def test_reading_json_survives_structured_finalization_without_prose_rewrite():
    raw = """```json
    {"what_i_learned":"A bounded finding","candidate_priors":[],"open_threads":[]}
    ```"""

    finalized = canonicalize_structured_output(raw)

    assert json.loads(finalized) == {
        "what_i_learned": "A bounded finding",
        "candidate_priors": [],
        "open_threads": [],
    }


def test_reading_structured_finalization_rejects_prose():
    with pytest.raises(ValueError):
        canonicalize_structured_output("I learned something, but this is prose.")


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
        }), None, _fetched(str(req.url)))
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
        return GenerateOutcome(json.dumps(response), None, _fetched(seed.url))
    pipe._generate = generate
    with pytest.raises(ValueError) as caught:
        asyncio.run(pipe._stage1_read(seed))
    # Rejected for its content, not for a missing fetch.
    assert not isinstance(caught.value, NoReadEvidenceError)


# Live 2026-09-25 hollow read (finding:60d59b10...:9b084fc0f1583da0): a
# well-formed, honest-sounding handoff from a turn that fetched nothing.
_LIVE_HOLLOW = {
    "what_i_learned": (
        "Metadata-only extraction; I did not fetch or read the article body this "
        "turn, and no claim about the page's content is grounded."
    ),
    "candidate_priors": [{"claim": "The URL is a hub page, inferred from its slug.", "confidence": 0.5}],
    "concept_candidates": [{"label": "network_world", "definition": "IT trade publication."}],
    "open_threads": ["Evidence gap: the article body was never fetched."],
}


@pytest.mark.parametrize("fetches,label", [
    ([], "no_read_evidence"),
    (None, "no_read_evidence:harness_unreported"),
    ([SourceFetchEvidenceV1(url="https://other.example.net/x", tool_name="WebFetch", content_chars=4000)], "no_read_evidence"),
])
def test_schema_valid_handoff_without_a_source_fetch_is_not_a_read(fetches, label):
    url = "https://www.networkworld.com/article/3562856/nvidia-latest-news-and-insights.html"
    seed = WorldPulseReadSeedV1(seed_id="finding:r:x", kind="finding", run_id="r", url=url)
    pipe = object.__new__(WorldPulseReadPipeline)
    async def generate(prompt, trace):
        return GenerateOutcome(json.dumps(_LIVE_HOLLOW), None, fetches)
    pipe._generate = generate
    with pytest.raises(NoReadEvidenceError) as caught:
        asyncio.run(pipe._stage1_read(seed))
    assert str(caught.value) == label
