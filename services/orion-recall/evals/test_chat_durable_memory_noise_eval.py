from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_chat_durable_memory_noise_eval", Path(__file__).with_name("run_chat_durable_memory_noise_eval.py")
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_classify_counts_each_noise_kind_and_keeps_real_chat_separate():
    rendered = "\n".join(
        [
            "[Recent thread — continuity only]",
            "Recent context:",
            "- [sql_chat:chat_history_log] Juniper: hi",
            "- [sql_chat:aitown_chat_history_log] Mara Vale: hello",
            "- [bus_synaptic_anomaly:falkordb] hop latency |z|=58.9",
            "- [active_packet] stale request",
            "- [concept_region] sub-concept-topicfoundry-x-4 associated_with sub-concept-seed-orion",
            "- [concept_region] Orion: the digital mind",
        ]
    )
    c = _MOD.classify(rendered)
    assert c["sql_chat:chat_history_log"] == 1
    assert c["sql_chat:aitown_chat_history_log"] == 1
    assert c["bus_synaptic_anomaly"] == 1
    assert c["active_packet"] == 1
    assert c["concept_region:topicfoundry"] == 1
    assert c["concept_region"] == 1
    assert sum(c[k] for k in _MOD.NOISE_KINDS) == 4
