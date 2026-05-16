"""orion-cortex-exec application package."""

from __future__ import annotations

# Phase-5 Mind/substrate convergence: install the shared cognitive projection
# spine before modules such as ``executor.py`` import chat stance helpers.
from .chat_stance_shared_spine import install_chat_stance_shared_spine

install_chat_stance_shared_spine()
