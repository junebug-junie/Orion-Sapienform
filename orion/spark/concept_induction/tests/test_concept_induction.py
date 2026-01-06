from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.clusterer import ConceptClusterer
from orion.spark.concept_induction.embedder import EmbeddingClient
from orion.spark.concept_induction.extractor import SpacyConceptExtractor
from orion.spark.concept_induction.inducer import ConceptInducer, WindowEvent
from orion.spark.concept_induction.settings import ConceptSettings


def _env_with_text(text: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="chat.message",
        source=ServiceRef(name="test", version="0.0.0"),
        payload={"content": text},
    )


class ConceptInductionTests(unittest.TestCase):
    def test_spacy_extraction_basic(self):
        extractor = SpacyConceptExtractor(model_name="en_core_web_sm")
        res = extractor.extract(["Orion talked with Juniper about concept induction."])
        self.assertTrue(any("orion" in c for c in res.candidates))

    def test_embedding_client_mock(self):
        calls = {}

        def _fake_post(url, json, timeout):
            calls["payload"] = json

            class Resp:
                def raise_for_status(self): ...

                def json(self):
                    return {"embeddings": {item: [1.0, 0.0] for item in json["items"]}}

            return Resp()

        with patch("requests.post", _fake_post):
            client = EmbeddingClient("http://fake")
            resp = client.embed(["alpha", "beta"])
            self.assertEqual(resp.embeddings["alpha"], [1.0, 0.0])
            self.assertIn("payload", calls)

    def test_clusterer_deterministic(self):
        clusterer = ConceptClusterer(threshold=0.5)
        res = clusterer.cluster(["orion self", "orion self", "juniper friend"], embeddings=None)
        self.assertEqual(len(res.clusters), 2)

    def test_profile_and_delta_generation(self):
        settings = ConceptSettings()
        inducer = ConceptInducer(settings)
        now = datetime.now(timezone.utc)
        window = [
            WindowEvent(text="Orion reflected with Juniper about the lake.", timestamp=now, envelope=_env_with_text("x")),
            WindowEvent(text="Juniper shared new plans for Orion", timestamp=now + timedelta(seconds=1), envelope=_env_with_text("y")),
        ]
        result = asyncio.run(inducer.run(subject="relationship", window=window))
        self.assertTrue(result.profile.concepts)
        result2 = asyncio.run(inducer.run(subject="relationship", window=window))
        self.assertGreaterEqual(result2.profile.revision, result.profile.revision)


if __name__ == "__main__":
    unittest.main()
