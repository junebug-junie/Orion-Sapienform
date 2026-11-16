"""The market's 30-day weights replay, run in a worker process (issue #3382, second cause).

The replay (``orion.physiology.weights``: ``weights_now`` twice plus ``endorsement_exclusions``) is pure-Python
CPU work, ~1.8-3.3 s on the live 30-day history. In a thread it held the GIL and starved the event loop that
answers ``POST /physiology/bid`` (callers wait 0.5 s). In a separate process it cannot.

Crossing the process boundary costs a pickle of the inputs. One ``pickle.dumps`` of the live history held the
GIL for ~85 ms in a single C call (2026-11-16: 5,292 acts, 5,286 uses, 8.6 MB, mostly ``cost_json`` text the
replay never reads). So ``pack`` runs in a worker thread, drops each act's ``cost`` (unread by the replay), and
pickles in chunks of ``CHUNK`` records: each chunk is a few milliseconds, and the GIL goes back to the bid loop
between chunks. The executor then only copies bytes. ``run_packed`` (in the worker process) unpacks and replays.

Module-level and import-light on purpose: a spawned worker imports only this module and the weights math.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import pickle
from concurrent.futures import Executor, ProcessPoolExecutor
from datetime import datetime
from typing import Any, Callable, Optional, Sequence

from orion.physiology.weights import endorsement_exclusions, weights_now

CHUNK = 250

Packed = tuple[bytes, tuple[bytes, ...], tuple[bytes, ...]]


def compute_weights(families: Any, acts: Sequence[Any], uses: Sequence[Any], now: datetime,
                    choices: Optional[list]) -> tuple[dict, dict, dict]:
    """(without source (c), with it, endorsement checks). Without choices both are the same dict."""
    base = weights_now(families, acts, uses, now)
    if choices is None:
        return base, base, {}
    checks = endorsement_exclusions(families, acts, choices, now)
    excluded = frozenset(f for f, c in checks.items() if c.excluded)
    with_c = weights_now(families, acts, uses, now, choices=choices, endorsement_excluded=excluded)
    return base, with_c, checks


def _chunks(seq: Sequence[Any]) -> tuple[bytes, ...]:
    return tuple(pickle.dumps(list(seq[i:i + CHUNK]), protocol=pickle.HIGHEST_PROTOCOL)
                 for i in range(0, len(seq), CHUNK))


def pack(families: Any, acts: Sequence[Any], uses: Sequence[Any], now: datetime,
         choices: Optional[list]) -> Packed:
    """Run in a worker thread. ``cost`` is dropped from each act: the replay never reads it (its test pins
    that the weights are the same with and without it)."""
    slim = [dataclasses.replace(a, cost={}) if a.cost else a for a in acts]
    head = pickle.dumps((families, now, choices), protocol=pickle.HIGHEST_PROTOCOL)
    return head, _chunks(slim), _chunks(uses)


def _unchunk(parts: Sequence[bytes]) -> list:
    out: list = []
    for p in parts:
        out.extend(pickle.loads(p))
    return out


def run_packed(replay: Callable[..., tuple[dict, dict, dict]], packed: Packed) -> tuple[dict, dict, dict]:
    """In the worker process: unpack and replay. ``replay`` is a module-level function (pickled by name)."""
    head, acts, uses = packed
    families, now, choices = pickle.loads(head)
    return replay(families, _unchunk(acts), _unchunk(uses), now, choices)


def process_executor() -> Executor:
    """One worker process. ``spawn``, not ``fork``: forking a process that runs an event loop and threads can
    copy a held lock into the child and deadlock it."""
    return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))


__all__ = ["CHUNK", "compute_weights", "pack", "process_executor", "run_packed"]
