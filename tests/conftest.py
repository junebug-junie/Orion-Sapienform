"""Repo-level test defaults.

The self-monitor act gate (orion/self_monitor/gate.py) is ON by default and reads the live organ
state from Redis ORION_BUS_URL. A unit test must never let live state decide what a worker admits,
so it is off here; gate tests build ActGate directly with an injected view.
"""
import os

os.environ["SELF_MONITOR_ACT_GATE_ENABLED"] = "false"
