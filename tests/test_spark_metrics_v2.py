import os
import time
import unittest
from datetime import datetime, timezone
import numpy as np

from orion.spark.orion_tissue import OrionTissue
from orion.spark.spark_engine import SparkEngine
from orion.schemas.telemetry.spark_signal import SparkSignalV1


class SparkMetricsV2Tests(unittest.TestCase):
    def test_novelty_baseline_and_spike(self):
        tissue = OrionTissue(H=2, W=2, C=2, novelty_window=10)
        stim = np.ones((2, 2, 2), dtype=np.float32) * 0.1

        novelties = []
        for _ in range(10):
            novelties.append(tissue.calculate_novelty(stim, channel_key="chat"))
            tissue.propagate(stim, steps=1, learning_rate=0.2, channel_key="chat")

        # Baseline novelty should remain non-zero and stable
        self.assertGreater(novelties[-1], 0.05)
        self.assertLess(novelties[-1], 1.0)

        # New stimulus should spike novelty relative to baseline
        stim2 = np.ones((2, 2, 2), dtype=np.float32) * 2.0
        spike = tissue.calculate_novelty(stim2, channel_key="chat")
        self.assertGreater(spike, novelties[-1])

    def test_adaptive_learning_rate_bounds(self):
        tissue = OrionTissue(H=2, W=2, C=1, novelty_window=5)
        stim = np.ones((2, 2, 1), dtype=np.float32)

        # Low coherence + distress should learn slowly
        tissue.T = np.zeros_like(tissue.T)
        tissue.expectations["chat"] = np.zeros_like(stim)
        low_embed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tissue.propagate(stim, steps=1, learning_rate=0.2, channel_key="chat", embedding=low_embed, distress=0.5)
        low_mean = tissue.expectations["chat"].mean()

        # High coherence + low distress should learn faster
        tissue.T = np.zeros_like(tissue.T)
        tissue.expectations["chat"] = np.zeros_like(stim)
        tissue.embedding_expectations["chat"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        high_embed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tissue.propagate(stim, steps=1, learning_rate=0.2, channel_key="chat", embedding=high_embed, distress=0.0)
        high_mean = tissue.expectations["chat"].mean()

        self.assertGreater(high_mean, low_mean)
        self.assertGreater(high_mean, 0.0)
        self.assertLessEqual(high_mean, stim.mean())

    def test_distress_signal_expires(self):
        engine = SparkEngine(H=4, W=4, C=2)
        sig = SparkSignalV1(
            signal_type="equilibrium",
            intensity=0.8,
            as_of_ts=datetime.now(timezone.utc),
            ttl_ms=50,
            source_service="test",
        )
        engine.apply_signal(sig)
        engine.record_chat("hi", agent_id="tester", tags=["chat"])
        self.assertGreater(engine._distress_level, 0.0)
        time.sleep(0.1)
        engine.record_chat("hi again", agent_id="tester", tags=["chat"])
        self.assertEqual(engine._distress_level, 0.0)


if __name__ == "__main__":
    unittest.main()
