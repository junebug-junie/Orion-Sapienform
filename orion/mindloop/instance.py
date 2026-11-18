"""What a loop instance is called: its identity on the bus, in the body economy, the ledger, Redis and the logs.

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md section 10. An instance is a subclass of
``orion.mindloop.runtime.MindLoopRuntime`` that sets ``identity`` and ``state_model`` and implements the hooks
documented on that class. Only what a second instance (the curiosity replica: one model call per plan request,
family ``curiosity.replica``, tier 2, lane chat) will reuse is here; everything else stays in the instance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopIdentity:
    """Who the loop is on the bus, in the body economy, in the act ledger, in Redis and in the logs.

    * ``name``: prefix of every log line (``<name>_thought``, ``<name>_try_no_model`` ...), of the want id
      (``<name>-think-<hex>``) and of the asyncio task (``orion-<name>``).
    * ``family`` / ``producer`` / ``tier``: the ``WantV1`` the loop bids with
      (config/physiology/families.yaml must declare the family).
    * ``actor`` / ``source_service`` / ``thought_component``: the thought act's actor and publish source.
    * ``thought_ref``: the self-monitor gate's object_ref for the thought (``<ref>`` then the thought id
      in the act's own object_ref).
    * ``state_key``: the Redis key the state is saved under. One key per instance; instances never share.
    * ``settings_prefix``: the instance's settings, e.g. ``PRIME_`` for ``PRIME_TICK_SEC``. The runtime reads
      ``TICK_SEC``, ``REFRACTORY_SEC``, ``BID_LANE``, ``LLM_ROUTE``, ``STARVED_LLM_ROUTE`` (optional),
      ``SEAT_WAIT_SEC`` (optional), ``LLM_TIMEOUT_SEC`` and ``MAX_TOKENS`` under it, at use time.
    """

    name: str
    family: str
    actor: str
    producer: str
    state_key: str
    settings_prefix: str
    source_service: str
    thought_component: str
    thought_ref: str
    tier: int
    logger_name: str
    verb: str = "infer"
    temperature: float = 0.4
    thinking: bool = False
    # The cost ceiling's basis says "... s per <cost_unit>".
    cost_unit: str = "thought"


__all__ = ["LoopIdentity"]
