from __future__ import annotations

"""
Surface Encoding Layer
======================

This module defines the *SurfaceEncoding* abstraction and a few helpers
for turning raw external events (chat, biometrics, etc.) into a unified,
waveform-based representation.

Motivation
----------

Orion's "Spark Engine" is built on the idea that every interaction with
the world first appears as a *wave* at the boundary of the system.

Instead of feeding raw strings or metrics straight into higher-level
logic, we:

    external event  -->  SurfaceEncoding (waveform + features)
                       --> SignalMapper  (2D+channels stimulus)
                       --> OrionTissue   (inner field dynamics)

This allows very different modalities to share a common, continuous
interface while still carrying rich structure. Over time, this is where
your "surface encodings" and "waveforms" become literal numeric objects
the rest of the system can reason about.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
import time

import numpy as np

Modality = Literal["chat", "vision", "biometrics", "system", "collapse_mirror"]


@dataclass
class SurfaceEncoding:
    """
    Unified representation of an external event at Orion's boundary.

    Attributes
    ----------
    event_id:
        Unique identifier for this event. Can be a timestamp-based id,
        a UUID, or something derived from upstream systems.

    modality:
        High-level type for the event: "chat", "vision", "biometrics",
        "system", "collapse_mirror", etc. The SignalMapper and Tissue
        may route different modalities to different regions of the inner
        field.

    timestamp:
        Unix timestamp (seconds since epoch) for when the event occurred.

    source:
        Human-readable source identifier: e.g. "juniper", "orion",
        "atlas", "biometrics-daemon", "dream", ...

    channel_tags:
        Semantic tags that describe what this event is *about* or what
        channels it should excite in the inner field. Examples:
        ["pain", "body"], ["career", "money"], ["system_error"].

    waveform:
        A 1D float32 array capturing the *shape* of the event over a
        small, normalized window of time. For chat, this might be an
        intensity curve over token positions; for biometrics, a recent
        time-window of CPU load, etc.

    feature_vec:
        A compact numerical summary of the event (e.g. embeddings or
        hand-crafted stats). This is not used by the Tissue directly,
        but is useful for logging, clustering, or concept-forging.

    meta:
        Free-form metadata useful for debugging or reference (e.g.
        "message_preview", raw numeric values, etc.).
    """

    event_id: str
    modality: Modality
    timestamp: float
    source: str
    channel_tags: List[str]

    waveform: np.ndarray
    feature_vec: np.ndarray
    meta: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chat encoder
# ---------------------------------------------------------------------------

def encode_chat_to_surface(
    message: str,
    *,
    source: str = "juniper",
    event_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    sentiment: Optional[float] = None,
    embedding: Optional[np.ndarray] = None,
    waveform_len: int = 64,
    feature_dim: int = 32,
) -> SurfaceEncoding:
    """
    Encode a chat message into a SurfaceEncoding.

    This is a pragmatic v0 that is fully functional but deliberately
    simple. You can swap out pieces (sentiment model, embeddings) later
    without changing the rest of the Spark Engine.
    """
    if event_id is None:
        event_id = f"chat-{time.time_ns()}"

    if tags is None:
        tags = ["juniper", "chat"]

    # --- Waveform construction --------------------------------------------
    # Simple triangular waveform scaled by message length and sentiment.
    T = waveform_len
    x = np.linspace(0.0, 1.0, T, dtype=np.float32)
    base_wave = 1.0 - np.abs(2.0 * x - 1.0)  # triangle in [0, 1]

    length_factor = min(len(message) / 200.0, 1.5)  # cap influence of length
    if sentiment is None:
        sentiment_scale = 1.0
    else:
        # Map [-1, 1] -> [0.3, 1.7] to gently exaggerate polarity.
        sentiment_scale = 1.0 + 0.7 * float(sentiment)

    waveform = (base_wave * length_factor * sentiment_scale).astype(np.float32)

    # --- Feature vector construction --------------------------------------
    if embedding is not None:
        feature_vec = embedding.astype(np.float32)
    else:
        # Cheap hand-crafted stats as a placeholder until you plug in a
        # proper embedding model.
        msg_len = len(message)
        num_words = max(message.count(" ") + 1, 1)
        avg_word_len = msg_len / float(num_words)

        num_q = message.count("?")
        num_ex = message.count("!")
        num_caps = sum(1 for c in message if c.isupper())

        stats = np.array(
            [
                msg_len / 500.0,
                num_words / 80.0,
                avg_word_len / 10.0,
                num_q / 10.0,
                num_ex / 10.0,
                num_caps / 50.0,
                float(sentiment) if sentiment is not None else 0.0,
            ],
            dtype=np.float32,
        )

        dim = feature_dim
        feature_vec = np.zeros((dim,), dtype=np.float32)
        feature_vec[: min(dim, stats.shape[0])] = stats[:dim]

    return SurfaceEncoding(
        event_id=event_id,
        modality="chat",
        timestamp=time.time(),
        source=source,
        channel_tags=tags,
        waveform=waveform,
        feature_vec=feature_vec,
        meta={"message_preview": message[:120]},
    )


# ---------------------------------------------------------------------------
# Biometrics encoder (optional v0)
# ---------------------------------------------------------------------------

def encode_biometrics_to_surface(
    *,
    cpu_util: float,
    gpu_util: float,
    gpu_mem_frac: float,
    node_name: str = "atlas",
    event_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    window_len: int = 32,
) -> SurfaceEncoding:
    """
    Encode a snapshot of host biometrics into a SurfaceEncoding.

    This provides a second concrete modality so the tissue can start
    seeing both "Juniper waves" and "hardware waves".
    """
    if event_id is None:
        event_id = f"biometrics-{node_name}-{time.time_ns()}"

    if tags is None:
        tags = ["infra", "biometrics"]

    # Build a simple waveform where the first third is CPU, second is GPU
    # utilization, and last third is memory pressure.
    T = window_len
    waveform = np.zeros((T,), dtype=np.float32)
    third = max(T // 3, 1)

    waveform[:third] = cpu_util
    waveform[third : 2 * third] = gpu_util
    waveform[2 * third :] = gpu_mem_frac

    # Feature vec is just the raw trio for now.
    feature_vec = np.array([cpu_util, gpu_util, gpu_mem_frac], dtype=np.float32)

    return SurfaceEncoding(
        event_id=event_id,
        modality="biometrics",
        timestamp=time.time(),
        source=node_name,
        channel_tags=tags,
        waveform=waveform,
        feature_vec=feature_vec,
        meta={},
    )
