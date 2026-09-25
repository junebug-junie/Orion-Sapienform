import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).with_name("run_inference_outcome_eval.py")
_spec = importlib.util.spec_from_file_location("inference_outcome_eval", _PATH)
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev
_spec.loader.exec_module(ev)

LOG = """\
2026-09-24T21:00:01.1Z [LLM-GW] INFO - gateway_llm_route_selected correlation_id=a route=quick served_by=circe-worker-fast-1 model=m
2026-09-24T21:00:02.1Z [LLM-GW] INFO - gateway_llm_route_selected correlation_id=b route=metacog served_by=circe-worker-2 model=m
2026-09-24T21:00:03.1Z [LLM-GW] ERROR - [LLM-GW] llamacpp TIMEOUT route=metacog served_by=circe-worker-2 url=u corr=c timeouts=t
2026-09-24T21:00:03.2Z [LLM-GW] INFO - gateway_llm_route_selected correlation_id=c route=metacog served_by=circe-worker-2 model=m
2026-09-24T21:00:04.1Z [LLM-GW] ERROR - [LLM-GW] llamacpp error: boom
2026-09-24T21:00:05.1Z [LLM-GW] WARNING - gateway_overloaded correlation_id=d stage=upstream_queue route=x
2026-09-24T21:01:01.1Z [LLM-GW] INFO - gateway_llm_route_selected correlation_id=e route=quick served_by=circe-worker-fast-1 model=m
unrelated line
"""


def test_replay_counts_failure_share_per_window_and_node():
    out = ev.replay(LOG.splitlines(), window_sec=60)
    circe = out["inference_failure_pressure_by_node"]["circe"]
    # window 21:00 -> 1 timeout of 3 replies; window 21:01 -> 0 of 1
    assert circe["windows_with_traffic"] == 2
    assert circe["windows_at_zero"] == 1
    assert abs(circe["max"] - 1 / 3) < 1e-9
    assert out["unattributed_backend_failure_lines"] == 1
    assert out["windows"] == 2


def test_live_rest_state_is_reachable_on_a_calm_log():
    calm = [l for l in LOG.splitlines() if "route_selected" in l]
    out = ev.replay(calm, window_sec=60)
    assert out["inference_failure_pressure_by_node"]["circe"]["max"] == 0.0
