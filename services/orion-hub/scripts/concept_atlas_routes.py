"""Concept Atlas — read-only interpretability routes for the substrate concept graph.

Phase 8 of the concept-graph-pipeline design
(``docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md``).

This module owns exactly two GET routes plus the standalone page route for the
iframe-embedded Concept Atlas Hub tab. It deliberately does not construct its
own substrate store instance: ``scripts.api_routes`` already builds one shared
``SUBSTRATE_SEMANTIC_STORE`` at import time via ``build_substrate_store_from_env()``
(see ``services/orion-hub/scripts/api_routes.py``), and ``memory_graph_routes.py``
already establishes the convention of importing ``scripts.api_routes`` inside
route functions (deferred import) rather than at module load time to reuse that
shared instance. This module follows the same convention.

If that store is ever not attached / not importable, every route here degrades
to an honest "unavailable" response instead of fabricating data or raising a
500 — this is a debug/interpretability surface, not a critical path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from .settings import settings
from .topic_foundry_client import (
    TopicFoundryClientError,
    create_dataset,
    create_model,
    fetch_latest_completed_run,
    fetch_mention_edges_for_run,
    fetch_run_topics_and_keywords,
    fetch_segments_for_run,
    list_datasets,
    list_models,
    trigger_enrichment_for_run,
    trigger_training_run,
)

logger = logging.getLogger("orion-hub.concept_atlas")

router = APIRouter(tags=["concept-atlas"])

# Cap on how many nodes get flagged "god node" in the network response. Every
# canonical (golden-seeded) node is always included regardless of this cap --
# see concept_atlas_network()'s canonical_ids handling -- and any remaining
# slots up to this total go to the highest (activation-weighted) degree
# non-canonical nodes. The store is small and in-memory (per the design
# spec's explicit non-goal of no precomputed ranking job), so this is
# computed fresh on every request rather than cached.
_GOD_NODE_TOP_N = 5
# orion/substrate/adapters/topic_foundry.py sets this as provenance.producer on
# every node it emits. Unlike metadata["source"], it is a native Cypher column
# and therefore survives rehydration -- see the "origin" comment below.
_TOPIC_FOUNDRY_PRODUCER = "topic_foundry_adapter"

_VALID_ANCHOR_SCOPES = {"orion", "juniper", "claude", "relationship", "world", "session"}

# Extra off-slice nodes concept_atlas_network() will hydrate via
# store.get_node_by_id() -- see that function's own comment for why. Bounds
# worst-case work the same "generous but bounded" way as this file's other
# caps (_GOD_NODE_TOP_N, the 300/600 node/edge limits below).
_NETWORK_HYDRATION_MAX_EXTRA_NODES = 100
# promotion_state is not a network-route query param per the design spec's
# route contract (scope/min_activation/focus only); the Concept Atlas UI
# applies its promotion_state global filter client-side against this
# endpoint's response instead of round-tripping it server-side.

# Concept nodes sitting within this margin of their own decay_floor are
# treated as "at risk" in the summary response.
_AT_RISK_MARGIN = 0.05

# A concept must be at least this old before it's eligible for "at risk" --
# see ``_at_risk_concepts`` for why (a node born with low salience hasn't
# decayed, it just started low). One hour comfortably exceeds one 120s decay
# tick interval.
_AT_RISK_MIN_AGE_SECONDS = 3600.0

# Max co_occurs_with pairs classified for a typed relationship (supports/
# contradicts/refines) per ingestion call. Each classified pair costs one
# bounded LLM RPC (see concept_relation_classifier.py), so this directly
# bounds the added route latency: worst case (every RPC timing out at
# HUB_LLM_GATEWAY_TIMEOUT_SEC, default 5s) is roughly
# _RELATION_CLASSIFICATION_PAIR_CAP * 5s of extra route time. Picked at the
# low end of the task's suggested 10-20 range for that reason -- this is a
# manual, operator-triggered route (not a hot path), but an unbounded or
# high cap could still make one ingest call take minutes.
_RELATION_CLASSIFICATION_PAIR_CAP = 10

# Well-known, fixed names for the scheduler's own topic-foundry dataset/model
# (Gap 5). Idempotent get-or-create by name -- see
# _ensure_topic_foundry_dataset_and_model -- so the scheduler survives Hub
# restarts and topic-foundry redeploys without losing track of "its"
# dataset/model. Same source-table/columns/windowing/model-spec shape as
# services/orion-topic-foundry/scripts/smoke_topic_foundry_train_and_poll.sh's
# defaults (the only other place this pipeline has been exercised end-to-end
# against real chat history).
#
# "-v2" (2026-08-18): renamed from "orion-hub-autonomous-dataset"/
# "orion-hub-autonomous" to force fresh creation with _TOPIC_FOUNDRY_WHERE_SQL
# applied. topic-foundry's dataset/model routes are create/list/preview only
# -- no update endpoint (services/orion-topic-foundry/app/routers/datasets.py,
# models.py) -- so changing where_sql on the *old* name would have been a
# silent no-op: _ensure_topic_foundry_dataset_and_model does get-or-create by
# name, and would keep finding and reusing the pre-existing, unfiltered
# dataset/model forever. See
# docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md
# ("Track A") for why this filter exists: chat_history_log is ~90%
# source='orion-embodiment' (AI Town) rows with no prior platform scoping,
# so Orion's "organically clustered" concept graph had been mostly AI Town
# topics since this pipeline first ran. The old, unfiltered dataset/model
# are left in place -- topic-foundry has no delete endpoint either -- and
# get-or-create-by-name (above) never looks them up again once this file's
# constants point at the "-v2" names. They were *not* fully inert until a
# second review-caught bug was fixed the same day: fetch_latest_completed_run
# (topic_foundry_client.py) and fetch_run_topics_and_keywords resolved "the
# latest run" globally across every model, not scoped to this one, so
# ingestion could keep silently reading the old model's runs regardless of
# this rename. Both call sites below now pass
# model_name=_TOPIC_FOUNDRY_MODEL_NAME.
#
# Model name history (2026-08-19): this file's model_spec had hardcoded
# min_cluster_size=15 / metric="euclidean" since the pipeline first shipped
# -- min_cluster_size=15 is flagged by topic-foundry's own 2026-07-21
# incident note (services/orion-topic-foundry/app/models.py::ModelSpec) as
# producing 1-2 degenerate clusters on a 676-document corpus, and produced
# 0 clusters every run on the real, much smaller "-v2" corpus (60-160ish
# documents after the AI Town filter, live-verified 2026-08-18/19). Fixed
# to read from settings (SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE/
# _METRIC, see app/settings.py for the full live-verification story,
# including a first attempt with metric="cosine" that failed outright --
# "Unrecognized metric 'cosine'" -- because the installed hdbscan library's
# real clusterer does not support it; corrected to "euclidean", confirmed
# live to actually cluster: cluster_count=3 on 62 real documents).
#
# The base name below is suffixed with a fingerprint of the exact settings
# that feed the model_spec (see _topic_foundry_model_spec_fingerprint) --
# NOT a hand-bumped "-v3"/"-v4" version suffix. Same create-only
# constraint as the dataset above (model_spec is fixed at creation,
# get-or-create is by name), but code review on this exact patch flagged
# that a hand-bumped suffix silently reproduces the same bug class it was
# introduced to fix: change SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_* without also
# remembering to bump the name, and get-or-create keeps training on the
# OLD model_spec forever with no warning (unlike the dataset's where_sql
# case just above, topic-foundry's GET /models returns ModelSummary, not
# model_spec, so there's nothing to compare against after creation to even
# detect the drift). Deriving the suffix from the settings themselves
# means a real config change always produces a new name automatically --
# the failure mode is structurally closed, not just documented. Dataset
# name is unchanged -- only the model_spec is settings-driven, the
# where_sql filter is not.
_TOPIC_FOUNDRY_DATASET_NAME = "orion-hub-autonomous-dataset-v2"
_TOPIC_FOUNDRY_MODEL_NAME_BASE = "orion-hub-autonomous-v4"
_TOPIC_FOUNDRY_MODEL_VERSION = "v1"


# Which dataset text column each speaker authored. `chat_history_log` stores
# one full exchange per row: `prompt` is always Juniper, `response` is always
# Orion. This is recorded fact, not inference -- confirmed live 2026-08-28,
# both speak in 254/254 rows while their *names* appear in only 28%/26%, which
# is why participation must come from here and not from entity extraction over
# the text. Values are lowercase to match `_speaker_concept_ids()`'s
# `label.lower() -> node_id` keys -- NOT `_landmark_concept_ids()`, which as of
# 2026-08-28 deliberately excludes exactly these speakers.
_TOPIC_FOUNDRY_COLUMN_SPEAKERS = {"prompt": "juniper", "response": "orion"}
# AI Town's corpus is agent-to-agent; its prompt/response authors are not
# Juniper and Orion and are not currently resolvable, so it splits columns
# (still correct -- two different speakers) but records no speaker rather than
# guessing one.
_TOPIC_FOUNDRY_AITOWN_COLUMN_SPEAKERS: dict[str, str] = {}


def _topic_foundry_windowing_spec(column_speakers: dict[str, str]) -> dict[str, Any]:
    """The windowing half of a model's frozen spec.

    `block_mode="rows"` + `split_text_columns=True` means one document per
    utterance. It replaced `turn_pairs` on 2026-08-28: on a source whose every
    row already holds a full prompt+response exchange, turn_pairs paired two
    *complete exchanges* and stamped one "User:" and the other "Assistant:",
    embedding two false role labels into the vectorized text. See
    docs/superpowers/specs/2026-08-28-concept-induction-topic-model-rebuild-design.md.
    """
    return {
        "block_mode": "rows",
        "split_text_columns": True,
        "column_speakers": dict(column_speakers),
        # Derived from column_speakers, never hardcoded. The old literal
        # ["user", "assistant"] was a trap once speakers became real: under
        # block_mode="turn_pairs" the role filter would match neither
        # "juniper" nor "orion" and silently drop every block. Empty (AI
        # Town, speakers unknown) is falsy, so the filter short-circuits and
        # nothing is dropped -- same as its long-standing inert behavior.
        "include_roles": sorted(set(column_speakers.values())),
        "time_gap_seconds": 900,
        "max_window_seconds": 7200,
        "min_blocks_per_segment": 1,
        "max_chars": 6000,
    }


def _topic_foundry_model_spec_fingerprint(windowing_spec: Optional[dict[str, Any]] = None) -> str:
    """Short, deterministic fingerprint of a model's frozen spec fields. See
    the "Model name history" comment above.

    `windowing_spec` joined the fingerprint on 2026-08-28. It was missing, and
    that was the same latent bug the HDBSCAN fields were added to close: a
    model row freezes its `windowing_spec` at creation and get-or-create
    matches purely by name, so changing windowing without changing the name
    left every future run training on the OLD windowing forever, with no
    warning and nothing to diff against (GET /models returns ModelSummary,
    which omits both specs). Confirmed live: the model in service on
    2026-08-28 still carried `block_mode: turn_pairs` frozen in its row.
    """
    fingerprint_input = "|".join(
        [
            str(settings.SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE),
            str(settings.SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC),
            str(settings.SUBSTRATE_TOPIC_FOUNDRY_EMBEDDING_URL),
            json.dumps(windowing_spec or {}, sort_keys=True, separators=(",", ":")),
        ]
    )
    return hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()[:8]


_TOPIC_FOUNDRY_WINDOWING_SPEC = _topic_foundry_windowing_spec(_TOPIC_FOUNDRY_COLUMN_SPEAKERS)
_TOPIC_FOUNDRY_MODEL_NAME = (
    f"{_TOPIC_FOUNDRY_MODEL_NAME_BASE}-{_topic_foundry_model_spec_fingerprint(_TOPIC_FOUNDRY_WINDOWING_SPEC)}"
)
_TOPIC_FOUNDRY_SOURCE_TABLE = "chat_history_log"
_TOPIC_FOUNDRY_ID_COLUMN = "correlation_id"
_TOPIC_FOUNDRY_TIME_COLUMN = "created_at"
_TOPIC_FOUNDRY_TEXT_COLUMNS = ["prompt", "response"]
# Value must match services/orion-recall/app/chat_source_tagging.py's
# AITOWN_TAG constant exactly -- that module is this repo's canonical
# ai-town platform-tagging signal (client_meta.external_room.platform), and
# this file deliberately does NOT cross-import it (orion-hub reaching into
# another service's app/ package would violate this repo's service-boundary
# convention -- CLAUDE.md section 5 -- and no other test/route in this repo
# does that either). test_topic_foundry_scheduler.py asserts this literal
# stays "aitown" so drift is at least test-visible, even without an
# import-time link. If AITOWN_TAG's value ever changes, this must change
# with it by hand.
_AITOWN_PLATFORM_TAG = "aitown"
# Excludes AI Town rows via that canonical tag, not the `source` column --
# confirmed live 2026-08-18 that source='orion-embodiment' is 100%
# correlated with this tag today, but the tag is the sanctioned signal the
# rest of the codebase (chat_source_tagging.py, the aitown crystallization
# gate) already builds around, and is the one that stays correct if a
# second AI-Town-adjacent producer service ever appears. `IS DISTINCT FROM`
# (not `!=`) so NULL client_meta / non-aitown rows are kept, not silently
# dropped by SQL's three-valued NULL comparison semantics.
_TOPIC_FOUNDRY_WHERE_SQL = (
    f"(client_meta -> 'external_room' ->> 'platform') IS DISTINCT FROM '{_AITOWN_PLATFORM_TAG}'"
)

# AI Town's own concept graph (docs/superpowers/specs/2026-08-18-aitown-
# concept-graph-split-and-atlas-readability-design.md, "AI Town's own
# concept graph"). Separate dataset/model/graph, same topic-foundry
# instance -- interpretability-only, never fed into concept_induced/
# chat_stance (that spec's Non-goals).
#
# No where_sql filter needed here, unlike the Orion dataset above: this
# reads from aitown_chat_history_log directly, which is already AI-Town-only
# by construction (orion-sql-writer routes each row to exactly one table by
# platform tag -- PR #1734, the table-split cutover -- not a filter on a
# shared table). If that table split is ever reverted, this dataset would
# need the mirror-image where_sql (`= 'aitown'` instead of `IS DISTINCT
# FROM`) -- not needed today.
_TOPIC_FOUNDRY_AITOWN_DATASET_NAME = "orion-hub-aitown-dataset-v1"
_TOPIC_FOUNDRY_AITOWN_MODEL_NAME_BASE = "orion-hub-aitown-v1"
# Same fingerprint fn/HDBSCAN settings as the Orion model above -- no
# AI-Town-specific tuning yet (real chat-volume difference noted in the
# design doc's "Missing questions", deliberately deferred rather than
# guessed at without live cluster-quality data to tune against).
_TOPIC_FOUNDRY_AITOWN_WINDOWING_SPEC = _topic_foundry_windowing_spec(
    _TOPIC_FOUNDRY_AITOWN_COLUMN_SPEAKERS
)
_TOPIC_FOUNDRY_AITOWN_MODEL_NAME = (
    f"{_TOPIC_FOUNDRY_AITOWN_MODEL_NAME_BASE}-"
    f"{_topic_foundry_model_spec_fingerprint(_TOPIC_FOUNDRY_AITOWN_WINDOWING_SPEC)}"
)
_TOPIC_FOUNDRY_AITOWN_SOURCE_TABLE = "aitown_chat_history_log"

# HDBSCAN's noise/outlier bucket (topic-foundry side). Never a real topic --
# same convention as orion/substrate/adapters/topic_foundry.py's own
# OUTLIER_TOPIC_ID, not imported from there since this module deliberately
# has no import-time dependency on the adapter (it's only imported lazily
# inside concept_atlas_ingest_topic_foundry()).
_OUTLIER_TOPIC_ID = -1

# Must match fetch_segments_for_run's own `limit` default (the API caps at
# 1000). Used only to tell whether a segment fetch came back full, i.e. the
# run may have more segments than the participation shares were computed over.
_SEGMENTS_FETCH_LIMIT = 1000


def _day_bucket_from_timestamp(value: Any) -> Optional[str]:
    """Parse a topic-foundry segment's ``start_at`` (a plain string over the
    wire, typically ISO-8601, e.g. ``"2026-07-15T10:23:00Z"`` or
    ``"...+00:00"``) into a UTC-day bucket key (``"2026-07-15"``).

    Used as the ``segment_topic_map`` grouping key -- ``SegmentRecord`` has
    no direct session/conversation id, so day-bucketing is the best
    available real proxy for "same conversation window." Never raises --
    any unparseable value degrades to ``None`` (caller skips it).
    """
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).date().isoformat()
    except (TypeError, ValueError):
        return None


def _ensure_topic_foundry_dataset_and_model(
    base_url: str,
    *,
    dataset_name: str = _TOPIC_FOUNDRY_DATASET_NAME,
    model_name: str = _TOPIC_FOUNDRY_MODEL_NAME,
    source_table: str = _TOPIC_FOUNDRY_SOURCE_TABLE,
    where_sql: Optional[str] = _TOPIC_FOUNDRY_WHERE_SQL,
    windowing_spec: dict[str, Any],
) -> Optional[tuple[str, str]]:
    """Idempotent get-or-create for a scheduler dataset+model, by name.

    Parameterized (2026-08-20) so a second dataset/model pair -- AI Town's
    own concept graph, see the module constants above -- can share this
    exact logic (idempotent get-or-create, the where_sql-drift warning, the
    fingerprinted model name) instead of a hand-duplicated copy. Every
    keyword default matches the pre-parameterization Orion behavior exactly,
    so the zero-arg call shape callers already use is unchanged.

    Returns ``(dataset_id, model_id)``, or ``None`` on any failure -- never
    raises. Safe to call on every scheduler tick: after the first tick
    creates both, every later tick finds them by name and reuses the same
    ids.
    """
    try:
        datasets = list_datasets(base_url)
        dataset = next((d for d in datasets if d.get("name") == dataset_name), None)
        if dataset is None:
            dataset = create_dataset(
                base_url,
                {
                    "name": dataset_name,
                    "source_table": source_table,
                    "id_column": _TOPIC_FOUNDRY_ID_COLUMN,
                    "time_column": _TOPIC_FOUNDRY_TIME_COLUMN,
                    "text_columns": _TOPIC_FOUNDRY_TEXT_COLUMNS,
                    "timezone": "UTC",
                    "where_sql": where_sql,
                },
            )
        else:
            # Get-or-create matches purely by name -- topic-foundry's
            # dataset routes are create/list/preview only, no update
            # endpoint, so where_sql/source_table edited here without also
            # bumping the dataset name silently keeps training against the
            # OLD filter/table forever (this is the exact bug the "-v2"
            # rename in this file's constants comment exists to fix once
            # already; code review 2026-08-18 flagged that nothing stops the
            # same mistake next time -- and code review 2026-08-20 found
            # this parameterization added source_table as a second drifting
            # field with no analogous check). Loud, not silent: log so a
            # future change here that forgets to rename shows up in the
            # scheduler's own logs on the very next tick, instead of only
            # being discoverable by noticing stale/wrong-corpus concepts
            # months later.
            if dataset.get("where_sql") != where_sql:
                logger.warning(
                    "topic_foundry_dataset_where_sql_drift dataset_name=%s expected=%r actual=%r "
                    "-- rename the dataset constant to force a fresh dataset with the new filter",
                    dataset_name,
                    where_sql,
                    dataset.get("where_sql"),
                )
            if dataset.get("source_table") != source_table:
                logger.warning(
                    "topic_foundry_dataset_source_table_drift dataset_name=%s expected=%r actual=%r "
                    "-- rename the dataset constant to force a fresh dataset against the new table",
                    dataset_name,
                    source_table,
                    dataset.get("source_table"),
                )
        dataset_id = str(dataset["dataset_id"])

        models = list_models(base_url)
        model = next((m for m in models if m.get("name") == model_name), None)
        if model is None:
            model = create_model(
                base_url,
                {
                    "name": model_name,
                    "version": _TOPIC_FOUNDRY_MODEL_VERSION,
                    "stage": "development",
                    "dataset_id": dataset_id,
                    "model_spec": {
                        "algorithm": "hdbscan",
                        "embedding_source_url": str(settings.SUBSTRATE_TOPIC_FOUNDRY_EMBEDDING_URL),
                        "min_cluster_size": settings.SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE,
                        "metric": settings.SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC,
                        "params": {},
                    },
                    # Required, never defaulted here. A model's name encodes a
                    # fingerprint OF THIS SPEC, and the row freezes it at
                    # creation -- so a default would let a caller passing the AI
                    # Town model_name mint a model whose name says AI Town
                    # windowing while its frozen row says Orion's. Model rows are
                    # create-only and GET /models omits both specs, so that
                    # mismatch would be permanent and undetectable (review
                    # finding, 2026-08-28).
                    "windowing_spec": windowing_spec,
                    "metadata": {},
                },
            )
        # No model_spec-drift warning analogous to the dataset's where_sql
        # one above, and none is needed: topic-foundry's GET /models list
        # route returns ModelSummary (app/models.py), which does not
        # include model_spec at all -- only DatasetSpec (returned by
        # GET /datasets) carries the field a drift check would need to
        # compare against, so detecting drift after the fact isn't
        # possible here without a GET /models/{model_id} call topic-foundry
        # doesn't expose. Instead, the model name's fingerprint suffix (see
        # module constants above) closes the failure mode structurally: a
        # real settings change always produces a new name, so get-or-create
        # can never find a stale model and silently reuse it.
        model_id = str(model["model_id"])
        return dataset_id, model_id
    except Exception as exc:
        logger.warning("topic_foundry_dataset_model_ensure_failed dataset_name=%s error=%s", dataset_name, exc)
        return None


def trigger_topic_foundry_training_run(
    *,
    dataset_name: str = _TOPIC_FOUNDRY_DATASET_NAME,
    model_name: str = _TOPIC_FOUNDRY_MODEL_NAME,
    source_table: str = _TOPIC_FOUNDRY_SOURCE_TABLE,
    where_sql: Optional[str] = _TOPIC_FOUNDRY_WHERE_SQL,
    windowing_spec: dict[str, Any] = _TOPIC_FOUNDRY_WINDOWING_SPEC,
    log_prefix: str = "topic_foundry",
) -> dict[str, Any]:
    """Scheduler entry point (Gap 5): ensure the scheduler's dataset/model
    exist, then trigger a training run for a rolling
    ``SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS``-day window ending now.

    Parameterized (2026-08-20) so AI Town's own dataset/model can share this
    exact trigger logic -- see ``_ensure_topic_foundry_dataset_and_model``'s
    docstring for why. Every keyword default matches the pre-
    parameterization Orion behavior, so the zero-arg call this scheduler
    already uses is unchanged. ``log_prefix`` only affects log message
    names, so the two dataset's log lines stay distinguishable in practice.

    Fire-and-forget from this function's perspective -- training runs as a
    background task on topic-foundry's side (see
    ``services/orion-topic-foundry/app/routers/runs.py::train_run_endpoint``);
    this function returns as soon as the run is queued (or immediately, if
    topic-foundry's own ``spec_hash`` dedup finds an identical run already
    exists). Ingesting the run's results once it completes is a separate
    step -- see ``concept_atlas_ingest_topic_foundry()`` -- deliberately not
    called from here, since the two are wired as two independent steps of
    the same scheduler tick in ``main.py``, not a blocking call chain.

    Never raises. Returns a summary dict, always including ``"triggered":
    bool``.
    """
    base_url = str(getattr(settings, "TOPIC_FOUNDRY_BASE_URL", "") or "").strip()
    if not base_url:
        return {"triggered": False, "reason": "topic_foundry_base_url_not_configured"}

    ids = _ensure_topic_foundry_dataset_and_model(
        base_url,
        dataset_name=dataset_name,
        model_name=model_name,
        source_table=source_table,
        where_sql=where_sql,
        windowing_spec=windowing_spec,
    )
    if ids is None:
        return {"triggered": False, "reason": "dataset_or_model_resolution_failed"}
    dataset_id, model_id = ids

    window_days = max(1, int(getattr(settings, "SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS", 30)))
    # Floor to a day boundary (UTC midnight), NOT datetime.now(timezone.utc)
    # verbatim. topic-foundry's own train-trigger route dedups by a spec_hash
    # computed over the exact start_at/end_at it receives (see
    # services/orion-topic-foundry/app/routers/runs.py::train_run_endpoint
    # and app/services/spec_hash.py) -- with microsecond-precision "now" on
    # every call, spec_hash would differ on literally every tick, so the
    # dedup this scheduler's safety argument depends on (re-triggering an
    # unchanged window is a cheap no-op) would never actually fire, and every
    # single tick would kick off a brand-new HDBSCAN training run regardless
    # of interval. Flooring to a day boundary means every tick within the
    # same UTC day computes the identical (start_at, end_at) pair, so
    # repeated ticks within a day correctly resolve to topic-foundry's
    # existing queued/running/complete run instead of enqueueing a duplicate.
    end_at = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_at = end_at - timedelta(days=window_days)

    try:
        run = trigger_training_run(
            base_url,
            model_id=model_id,
            dataset_id=dataset_id,
            start_at=start_at.isoformat(),
            end_at=end_at.isoformat(),
        )
        return {
            "triggered": True,
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "dataset_id": dataset_id,
            "model_id": model_id,
            "window_days": window_days,
        }
    except TopicFoundryClientError as exc:
        logger.warning("%s_train_trigger_failed error=%s", log_prefix, exc)
        return {"triggered": False, "reason": "train_trigger_failed", "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive, never let a client bug crash the scheduler
        logger.warning("%s_train_trigger_unexpected_error error=%s", log_prefix, exc)
        return {"triggered": False, "reason": "unexpected_error", "error": str(exc)}


def trigger_topic_foundry_enrichment(
    *,
    model_name: str = _TOPIC_FOUNDRY_MODEL_NAME,
    log_prefix: str = "topic_foundry",
) -> dict[str, Any]:
    """Scheduler entry point, added 2026-07-28: trigger enrichment for
    whatever the latest COMPLETED run currently is.

    Confirmed live 2026-07-28: 0 of 22 real `topic_foundry_segments` rows had
    ever been enriched -- nothing in this codebase had ever called
    `POST /runs/{run_id}/enrich`. That endpoint also generates topic-foundry's
    typed KG edges as a same-request side effect on its side (see
    `app/services/enrichment.py::_run_enrichment`'s trailing
    `_generate_edges` call), which is what
    `orion/substrate/adapters/topic_foundry.py`'s mention-edge -> EntityNodeV1
    wiring depends on having real data to ingest.

    Gated by its own flag (`SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE`), separate
    from `SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED` -- real LLM compute cost
    per un-enriched segment, so it gets its own explicit go-ahead rather than
    riding on "the scheduler is already on."

    Fire-and-forget, same shape as `trigger_topic_foundry_training_run()`
    above: enrichment runs as a background task on topic-foundry's side, so
    this returns as soon as it's queued. The run enriched here is very
    plausibly a prior tick's completed run, not the one just triggered by
    `trigger_topic_foundry_training_run()` in this same tick (training takes
    real time) -- that's expected, matching the scheduler's existing
    "may act on a previous tick's run" pattern for ingestion.

    Never raises. Returns a summary dict, always including `"triggered": bool`.
    """
    if not bool(getattr(settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", False)):
        return {"triggered": False, "reason": "enrich_disabled"}

    base_url = str(getattr(settings, "TOPIC_FOUNDRY_BASE_URL", "") or "").strip()
    if not base_url:
        return {"triggered": False, "reason": "topic_foundry_base_url_not_configured"}

    try:
        # model_name-scoped: without it this can resolve to a *different*
        # model's latest run (e.g. the old, unfiltered "orion-hub-autonomous"
        # model, still live and still ticking) -- code review 2026-08-18.
        run = fetch_latest_completed_run(base_url, model_name=model_name)
    except TopicFoundryClientError as exc:
        logger.info("%s_enrich_no_completed_run reason=%s", log_prefix, exc)
        return {"triggered": False, "reason": "no_completed_run"}
    except Exception as exc:  # pragma: no cover - defensive, never let a client bug crash the scheduler
        logger.warning("%s_enrich_latest_run_lookup_unexpected_error error=%s", log_prefix, exc)
        return {"triggered": False, "reason": "unexpected_error", "error": str(exc)}

    run_id = run.get("run_id")
    if not run_id:
        return {"triggered": False, "reason": "no_completed_run"}

    # Review-caught 2026-07-28: `int(x or 0) or None` turns a deliberately
    # configured `SUBSTRATE_TOPIC_FOUNDRY_ENRICH_LIMIT=0` into `None` (no
    # cap sent at all) -- exactly backwards from what an operator setting it
    # to 0 almost certainly means ("pause without touching the enable
    # flag"). The server can't express "cap to exactly 0" either
    # (`_run_enrichment`'s own `if limit:` treats 0 as falsy = unlimited,
    # same bug shape one layer down) -- so 0/negative is handled here by not
    # calling the endpoint at all, the only way to honor "process nothing".
    raw_limit = int(getattr(settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_LIMIT", 200) or 0)
    if raw_limit <= 0:
        return {"triggered": False, "reason": "enrich_limit_non_positive", "run_id": run_id}
    limit = raw_limit
    try:
        result = trigger_enrichment_for_run(base_url, str(run_id), limit=limit, force=False)
        return {
            "triggered": True,
            "run_id": run_id,
            "status": result.get("status"),
            "enriched_count": result.get("enriched_count"),
            "failed_count": result.get("failed_count"),
        }
    except TopicFoundryClientError as exc:
        logger.warning("%s_enrich_trigger_failed run_id=%s error=%s", log_prefix, run_id, exc)
        return {"triggered": False, "reason": "enrich_trigger_failed", "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive, never let a client bug crash the scheduler
        logger.warning("%s_enrich_trigger_unexpected_error run_id=%s error=%s", log_prefix, run_id, exc)
        return {"triggered": False, "reason": "unexpected_error", "error": str(exc)}


def trigger_topic_foundry_aitown_training_run() -> dict[str, Any]:
    """AI Town's own concept graph -- zero-arg wrapper binding
    ``trigger_topic_foundry_training_run`` to the AI Town dataset/model
    constants above, so ``main.py``'s scheduler can call this exactly like
    the Orion step (``asyncio.to_thread(fn)``, no cross-module reach into
    this module's private constants needed at the call site)."""
    return trigger_topic_foundry_training_run(
        dataset_name=_TOPIC_FOUNDRY_AITOWN_DATASET_NAME,
        model_name=_TOPIC_FOUNDRY_AITOWN_MODEL_NAME,
        source_table=_TOPIC_FOUNDRY_AITOWN_SOURCE_TABLE,
        where_sql=None,
        windowing_spec=_TOPIC_FOUNDRY_AITOWN_WINDOWING_SPEC,
        log_prefix="topic_foundry_aitown",
    )


def trigger_topic_foundry_aitown_enrichment() -> dict[str, Any]:
    """Same as ``trigger_topic_foundry_aitown_training_run`` above, for
    ``trigger_topic_foundry_enrichment``."""
    return trigger_topic_foundry_enrichment(
        model_name=_TOPIC_FOUNDRY_AITOWN_MODEL_NAME,
        log_prefix="topic_foundry_aitown",
    )


def _get_named_substrate_store(attr_name: str, *, log_prefix: str) -> Any:
    """Best-effort resolution of a store singleton off ``scripts.api_routes``,
    never raises. Shared by ``_get_substrate_store``/``_get_aitown_substrate_store``
    below (review finding 2026-08-20: these were a hand-duplicated copy of
    each other, differing only in attribute name and log prefix).

    Deferred import mirrors ``memory_graph_routes.py``'s established pattern
    (``import scripts.api_routes as api_mod`` inside route functions) so this
    module does not force a heavy import of ``scripts.api_routes`` at module
    load time -- useful for isolated router tests, and consistent with the
    existing convention rather than inventing a new one.
    """
    try:
        import scripts.api_routes as api_mod
    except Exception as exc:  # pragma: no cover - defensive, mirrors route-level degrade
        logger.warning("%s_import_failed error=%s", log_prefix, exc)
        return None
    store = getattr(api_mod, attr_name, None)
    if store is None:
        logger.info("%s_not_attached", log_prefix)
    return store


def _get_substrate_store() -> Any:
    """Resolve the shared Orion substrate store (``api_routes.py``'s
    ``SUBSTRATE_SEMANTIC_STORE`` singleton). Kept as its own top-level
    function (not inlined at call sites) so tests can monkeypatch it
    directly, same as before this was factored through
    ``_get_named_substrate_store``.
    """
    return _get_named_substrate_store("SUBSTRATE_SEMANTIC_STORE", log_prefix="concept_atlas_store")


def _get_aitown_substrate_store() -> Any:
    """Same as ``_get_substrate_store`` above, for AI Town's own concept
    graph (``api_routes.py``'s ``SUBSTRATE_SEMANTIC_STORE_AITOWN`` singleton)."""
    return _get_named_substrate_store(
        "SUBSTRATE_SEMANTIC_STORE_AITOWN", log_prefix="concept_atlas_aitown_store"
    )


def _unavailable(reason: str, error: Optional[str] = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": False, "reason": reason}
    if error:
        payload["error"] = error
    payload.update(extra)
    return payload


class _CountingSubstrateStore:
    """Thin store wrapper that counts successful concept/evidence/edge upserts.

    ``SubstrateGraphMaterializer.apply_record`` writes incrementally and raises
    without returning a partial ``MaterializationResultV1``. Delegating through
    this wrapper lets the ingest route report precise successful counts on
    mid-record failure, while preserving all other store operations (including
    identity-resolver reads) via ``__getattr__``.
    """

    # An unbroken run of this many node-write failures is treated as the store
    # being down rather than the nodes being bad, and the exception is allowed
    # to propagate. Deliberately not tuned to separate "bad kind" from "dead
    # store" by counting alone -- the successful-write counter reset is what
    # does that. This is only a bound on how long a dead store can be retried.
    MAX_CONSECUTIVE_NODE_FAILURES = 10

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.concepts_written = 0
        self.evidence_nodes_written = 0
        self.entities_written = 0
        self.edges_written = 0
        # One node kind the store cannot persist used to abort the whole run.
        # Live 2026-08-29: the store raised on the first EntityNodeV1 and the
        # ingest ended at concepts_written=18 entities_written=0
        # edges_written=0 -- 18 orphaned Evidence nodes and not one edge, on
        # both the substrate and AI Town graphs. Entity nodes are durable as
        # of the same patch (falkor_codec.DURABLE_NODE_KINDS), so this is the
        # second, independent half: no single unwritable node may cost the
        # run its edges again.
        #
        # Skips are COUNTED AND SURFACED, never swallowed -- the ingest
        # response carries skipped_nodes/skipped_edges and the caller logs
        # them. A silent partial write reported as success is the failure
        # mode this whole route already exists to avoid.
        self.skipped_nodes: list[dict[str, Any]] = []
        self.skipped_edges = 0
        self._failed_node_ids: set[str] = set()
        self._consecutive_failures = 0

    @property
    def wrote_anything(self) -> bool:
        return bool(
            self.concepts_written
            or self.evidence_nodes_written
            or self.entities_written
            or self.edges_written
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: Any,
        skip_metadata_keys: Any = None,
    ) -> None:
        kind = getattr(node, "node_kind", None)
        try:
            self._inner.upsert_node(identity_key=identity_key, node=node, skip_metadata_keys=skip_metadata_keys)
        except Exception as exc:  # noqa: BLE001 - one bad node must not cost the run its edges
            node_id = str(getattr(node, "node_id", "") or "")
            # Only a REAL id joins the skip set. An empty id would otherwise sit
            # in it as a sentinel and match any later edge whose endpoint ref is
            # missing, dropping that edge and blaming `endpoint_not_written`
            # for an unrelated cause. Schema-typed SubstrateEdgeV1 refs are
            # min_length=3 so this cannot misfire today, but the wrapper is
            # duck-typed and a sentinel comparison is the wrong shape regardless.
            if node_id:
                self._failed_node_ids.add(node_id)
            self.skipped_nodes.append({"node_id": node_id, "node_kind": kind, "error": str(exc)})
            self._consecutive_failures += 1
            logger.warning(
                "concept_atlas_ingest_node_skipped node_id=%s node_kind=%s error=%s",
                node_id, kind, exc,
            )
            # RESILIENCE IS FOR A BAD NODE, NOT A DEAD STORE. Swallowing every
            # failure turns an unreachable FalkorDB into N caught exceptions, a
            # normally-completing apply_record, and a route that answers
            # `available: true` with every count at zero -- strictly worse than
            # the abort it replaced, and indistinguishable in the scheduler's
            # tick log from "no new data". It also costs N x connect-timeout.
            # An unbroken run of failures is a store problem, not a node
            # problem, so past this many the exception is allowed to propagate
            # to the route's _unavailable("substrate_store_write_failed") as
            # before. A single unwritable kind resets the counter on the next
            # success and never reaches it.
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_NODE_FAILURES:
                logger.error(
                    "concept_atlas_ingest_aborted_consecutive_failures count=%d last_error=%s",
                    self._consecutive_failures, exc,
                )
                raise
            return
        self._consecutive_failures = 0
        if kind == "concept":
            self.concepts_written += 1
        elif kind == "evidence":
            self.evidence_nodes_written += 1
        elif kind == "entity":
            self.entities_written += 1

    def upsert_edge(self, *, identity_key: str, edge: Any) -> None:
        # An edge whose endpoint was skipped must be skipped too. upsert_edge's
        # Cypher opens `MERGE (source:SubstrateNode {node_id: $source_id})`,
        # which CREATES a bare node when none exists -- so writing this edge
        # anyway would replace a skipped node with a phantom carrying nothing
        # but a node_id, which every reader would then decode as None. Dropping
        # the edge keeps the graph honest about what is missing.
        source_id = str(getattr(getattr(edge, "source", None), "node_id", "") or "")
        target_id = str(getattr(getattr(edge, "target", None), "node_id", "") or "")
        if source_id in self._failed_node_ids or target_id in self._failed_node_ids:
            self.skipped_edges += 1
            logger.warning(
                "concept_atlas_ingest_edge_skipped source=%s target=%s predicate=%s reason=endpoint_not_written",
                source_id, target_id, getattr(edge, "predicate", None),
            )
            return
        self._inner.upsert_edge(identity_key=identity_key, edge=edge)
        self.edges_written += 1


def _concept_nodes(store: Any) -> list[Any]:
    """All concept-kind nodes from the store's snapshot, or [] on any failure."""
    try:
        snapshot = store.snapshot()
    except Exception as exc:
        logger.warning("concept_atlas_snapshot_failed error=%s", exc)
        return []
    return [n for n in snapshot.nodes.values() if getattr(n, "node_kind", None) == "concept"]


def _typed_relation_classification_candidates(store: Any) -> tuple[dict[str, Any], list[Any]]:
    """Current concept nodes (by node_id) and not-yet-classified ``co_occurs_with``
    edges between them, read fresh from the store's snapshot. Returns ``({}, [])``
    on any store failure -- never raises.

    Excludes any ``co_occurs_with`` edge that already produced a typed edge in
    a prior pass -- ``classify_relation()`` stamps the typed edge's
    ``metadata["source_edge_id"]`` with the ``co_occurs_with`` edge's own
    ``edge_id`` (``orion/substrate/relation_classification.py``), so that
    marker is how "already classified" is recognized here. Without this
    filter, every ingestion call would re-spend its entire
    ``_RELATION_CLASSIFICATION_PAIR_CAP``-pair budget reclassifying pairs it
    already has an answer for (co-occurrence count only grows, so a
    qualifying pair keeps qualifying forever), permanently starving any
    candidate pair beyond the cap once the corpus grows past it. A pair whose
    prior judgment was "none" (no typed edge written) is deliberately NOT
    excluded -- its co-occurrence evidence keeps strengthening over time, so
    retrying it on a later pass is a reasonable, cheap use of a cap slot.
    """
    try:
        snapshot = store.snapshot()
    except Exception as exc:
        logger.warning("concept_atlas_relation_classification_snapshot_failed error=%s", exc)
        return {}, []
    concept_nodes = {
        n.node_id: n for n in snapshot.nodes.values() if getattr(n, "node_kind", None) == "concept"
    }
    already_classified_source_edge_ids = {
        e.metadata.get("source_edge_id")
        for e in snapshot.edges.values()
        if isinstance(e.metadata, dict) and e.metadata.get("source_edge_id")
    }
    co_occurs_edges = [
        e
        for e in snapshot.edges.values()
        if e.predicate == "co_occurs_with"
        and e.source.node_id in concept_nodes
        and e.target.node_id in concept_nodes
        and e.edge_id not in already_classified_source_edge_ids
    ]
    return concept_nodes, co_occurs_edges


def _classify_typed_concept_relations(store: Any) -> int:
    """Post-ingestion step (Phase 4 of the concept-graph-pipeline design):
    for ``co_occurs_with`` edges among current concept nodes that clear the
    ``count`` worth-classifying threshold
    (``orion.substrate.relation_classification.is_worth_classifying``,
    capped at ``_RELATION_CLASSIFICATION_PAIR_CAP`` pairs), ask a real LLM
    classifier whether the pair is a typed ``supports``/``contradicts``/
    ``refines`` relationship, and write any resulting typed edge into the
    same store.

    Additive best-effort enrichment, not required for the route's core
    success -- returns ``0`` (an honest count, not an error) on any failure
    or when nothing was worth classifying. Never raises past this function.
    """
    from orion.substrate.relation_classification import classify_relation, is_worth_classifying

    from .concept_relation_classifier import build_llm_relation_classifier

    try:
        concept_nodes, co_occurs_edges = _typed_relation_classification_candidates(store)
        if not co_occurs_edges:
            return 0

        candidate_pairs: list[tuple[Any, Any, Any]] = []
        for edge in co_occurs_edges:
            node_a = concept_nodes.get(edge.source.node_id)
            node_b = concept_nodes.get(edge.target.node_id)
            if node_a is None or node_b is None:
                continue
            # "count" strategy: the simplest baseline, matches the design
            # spec's called-out safe default. pmi/activation strategies exist
            # in relation_classification.py but wiring all three behind a
            # live flag is out of scope for this patch.
            if not is_worth_classifying(node_a, node_b, edge, strategy="count"):
                continue
            candidate_pairs.append((node_a, node_b, edge))
            if len(candidate_pairs) >= _RELATION_CLASSIFICATION_PAIR_CAP:
                break

        if not candidate_pairs:
            return 0

        classifier = build_llm_relation_classifier(candidate_pairs, settings=settings)

        typed_edges_written = 0
        for node_a, node_b, edge in candidate_pairs:
            new_edge = classify_relation(node_a, node_b, edge, classifier=classifier, strategy="count")
            if new_edge is None:
                continue
            try:
                identity_key = f"{new_edge.source.node_id}|{new_edge.predicate}|{new_edge.target.node_id}"
                store.upsert_edge(identity_key=identity_key, edge=new_edge)
                typed_edges_written += 1
            except Exception as exc:
                logger.warning("concept_atlas_typed_edge_write_failed error=%s", exc)
        return typed_edges_written
    except Exception as exc:  # pragma: no cover - defensive, mirrors this module's degrade-never-500 convention
        logger.warning("concept_atlas_relation_classification_failed error=%s", exc)
        return 0


def _at_risk_concepts(
    concept_nodes: list[Any], *, now: Optional[datetime] = None
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Concepts whose activation is decaying toward their decay_floor.

    Honesty note: this used to gate on "does activation show any variance
    across nodes" as a proxy for "is a live decay writer actually wired" --
    back when every ConceptNodeV1 was born with the exact same schema
    default (activation=0.0, decay_half_life_seconds=None), same-value
    across the board really did mean no real signal existed yet. Two fixes
    landed since (services/orion-hub/scripts/api_routes.py::decay_concept_activations,
    a live 120s scheduler; and ConceptNodeV1's own activation=salience
    auto-seed at construction time) mean activation is now a real signal
    from the moment a node is created, so the variance proxy is retired --
    it would otherwise misfire the other way, hiding genuinely-at-risk nodes
    the instant any two concepts happened to share a salience value.

    What replaces it: a concept born with low salience is not yet
    meaningfully "at risk of decaying" -- it just started low, it hasn't
    lost anything. So nodes younger than ``_AT_RISK_MIN_AGE_SECONDS`` (one
    hour -- comfortably more than one 120s decay tick, giving real decay a
    chance to actually run) are excluded regardless of how low their
    activation already is. This is still a real, non-fabricated filter, not
    a returned-empty placeholder.
    """
    if not concept_nodes:
        return [], None

    reference = now or datetime.now(timezone.utc)
    at_risk: list[dict[str, Any]] = []
    eligible_count = 0
    for n in concept_nodes:
        act = n.signals.activation
        observed_at = getattr(getattr(n, "temporal", None), "observed_at", None)
        if observed_at is None:
            continue
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        age_seconds = (reference - observed_at).total_seconds()
        if age_seconds < _AT_RISK_MIN_AGE_SECONDS:
            continue
        eligible_count += 1
        if float(act.activation) <= float(act.decay_floor) + _AT_RISK_MARGIN:
            at_risk.append(
                {
                    "node_id": n.node_id,
                    "label": getattr(n, "label", n.node_id),
                    "activation": float(act.activation),
                    "decay_floor": float(act.decay_floor),
                    "promotion_state": n.promotion_state,
                }
            )
    at_risk.sort(key=lambda row: row["activation"])

    note = None
    if not at_risk and eligible_count == 0:
        note = (
            f"no concept node is older than {int(_AT_RISK_MIN_AGE_SECONDS)}s yet -- "
            "at_risk is intentionally empty rather than judging a node's real decay "
            "before it has had a chance to run"
        )
    return at_risk[:20], note


_VALID_GRAPH_PARAM_VALUES = {"orion", "aitown"}


def _resolve_store_for_graph_param(graph: Optional[str]) -> tuple[Any, str]:
    """Resolve which store a request wants via the ``graph`` query param.

    Review finding 2026-08-20: before this, AI Town's own concept graph
    (``SUBSTRATE_SEMANTIC_STORE_AITOWN``) was written to every scheduler
    tick but reachable by zero GET route -- data written, structurally
    unreachable. This is the "first cut" the design spec's own "Missing
    questions" section named as sufficient before a dedicated AI Town
    Concept Atlas page is worth building: a reused
    ``?graph=aitown``-style parameter on the existing routes.

    Defaults to Orion's store for any unset/unrecognized value (never
    raises, matching this file's existing malformed-query-param handling
    for ``scope``/``min_activation`` in ``concept_atlas_network()`` below)
    -- an unrecognized ``graph`` value degrades to the default rather than
    500ing or silently returning nothing.

    Returns ``(store, resolved_graph_label)`` so callers can echo which
    graph actually served the response.
    """
    graph_norm = graph.strip().lower() if isinstance(graph, str) else ""
    if graph_norm == "aitown":
        return _get_aitown_substrate_store(), "aitown"
    if graph_norm and graph_norm not in _VALID_GRAPH_PARAM_VALUES:
        logger.info("concept_atlas_ignored_bad_graph_param value=%s", graph_norm)
    return _get_substrate_store(), "orion"


@router.get("/api/substrate/concepts/summary")
async def concept_atlas_summary(graph: Optional[str] = Query(None)) -> dict[str, Any]:
    store, graph_label = _resolve_store_for_graph_param(graph)
    if store is None:
        return _unavailable(
            "substrate_store_unavailable",
            total_concepts=0,
            by_promotion_state={},
            by_anchor_scope={},
            edge_counts_by_predicate={},
            at_risk=[],
            graph=graph_label,
        )

    concept_nodes = _concept_nodes(store)

    by_promotion_state: dict[str, int] = {}
    by_anchor_scope: dict[str, int] = {}
    for node in concept_nodes:
        by_promotion_state[node.promotion_state] = by_promotion_state.get(node.promotion_state, 0) + 1
        by_anchor_scope[node.anchor_scope] = by_anchor_scope.get(node.anchor_scope, 0) + 1

    edge_counts_by_predicate: dict[str, int] = {}
    try:
        snapshot = store.snapshot()
        concept_ids = {n.node_id for n in concept_nodes}
        for edge in snapshot.edges.values():
            if edge.source.node_id in concept_ids or edge.target.node_id in concept_ids:
                edge_counts_by_predicate[edge.predicate] = edge_counts_by_predicate.get(edge.predicate, 0) + 1
    except Exception as exc:
        logger.warning("concept_atlas_summary_edge_scan_failed error=%s", exc)

    at_risk, at_risk_note = _at_risk_concepts(concept_nodes)

    return {
        "available": True,
        "total_concepts": len(concept_nodes),
        "by_promotion_state": by_promotion_state,
        "by_anchor_scope": by_anchor_scope,
        "edge_counts_by_predicate": edge_counts_by_predicate,
        "at_risk": at_risk,
        "at_risk_note": at_risk_note,
        "graph": graph_label,
    }


def _display_labels(nodes: list[Any], edges: list[Any]) -> dict[str, str]:
    """A readable label for every node in the network view.

    Concepts already carry one. Evidence nodes do NOT -- ``EvidenceNodeV1``
    has no ``label`` field at all, so the payload fell back to the raw
    node_id and the atlas rendered rows of
    ``sub-evidence-topicfoundry-<uuid>-<n>``. They are still worth showing
    (they are what backs a concept), so name them after the concept they
    support, which is exactly what a reader wants to know about them.

    Falls back to the node_id when there is nothing better -- never empty.
    """
    labels: dict[str, str] = {}
    for node in nodes:
        raw = getattr(node, "label", None)
        labels[node.node_id] = str(raw) if raw else str(node.node_id)

    # `supports` runs evidence -> concept (see the topic_foundry adapter), so
    # the target is the concept whose name the evidence should borrow.
    concept_ids = {n.node_id for n in nodes if getattr(n, "node_kind", None) == "concept"}
    for edge in edges:
        if getattr(edge, "predicate", None) != "supports":
            continue
        source_id = getattr(edge.source, "node_id", None)
        target_id = getattr(edge.target, "node_id", None)
        if source_id is None or target_id not in concept_ids:
            continue
        if source_id in concept_ids:
            continue  # concept -> concept `supports` keeps its own label
        # Only fill in for a node that has NO label of its own. Guarding on
        # "not a concept" alone was too broad: EntityNodeV1 carries a real
        # label (min_length=1) and entity nodes are hydrated into this same
        # list, so the first producer to emit an entity -> concept `supports`
        # edge would silently rename a node called "Juniper" to
        # "Evidence for <topic>".
        if labels.get(source_id) != source_id:
            continue
        supported = labels.get(target_id)
        if supported:
            labels[source_id] = f"Evidence for {supported}"
    return labels


def _compute_connected_components(nodes: list[Any], edges: list[Any]) -> dict[str, int]:
    """Assign each node id a 0-indexed connected-component id via plain
    union-find over the (already filtered) node/edge lists.

    Readability gap named in
    ``docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md``:
    cose's force layout pulls disconnected components toward each other with
    nothing marking where one ends and another begins once the graph gets
    dense (``componentSpacing`` in concept-atlas.js only makes that less bad,
    it doesn't label components). This doesn't touch the layout -- it hands
    the frontend the grouping so it can (e.g. per-component styling, or a
    plain node count per component in the inspector).

    Component ids are numbered in the order their first member appears in
    ``nodes`` (not by root node id), so the numbering is deterministic given
    the same input ordering -- ``query_concept_region()`` already returns a
    stable order, this just doesn't add its own randomness on top.
    """
    parent: dict[str, str] = {n.node_id: n.node_id for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in edges:
        if e.source.node_id in parent and e.target.node_id in parent:
            union(e.source.node_id, e.target.node_id)

    component_id_by_root: dict[str, int] = {}
    result: dict[str, int] = {}
    for n in nodes:
        root = find(n.node_id)
        if root not in component_id_by_root:
            component_id_by_root[root] = len(component_id_by_root)
        result[n.node_id] = component_id_by_root[root]
    return result


@router.get("/api/substrate/concepts/network")
async def concept_atlas_network(
    scope: Optional[str] = Query(None),
    min_activation: Optional[str] = Query(None),
    focus: Optional[str] = Query(None),
    graph: Optional[str] = Query(None),
) -> dict[str, Any]:
    store, graph_label = _resolve_store_for_graph_param(graph)
    if store is None:
        return _unavailable(
            "substrate_store_unavailable",
            nodes=[],
            edges=[],
            god_node_count=0,
            component_count=0,
            graph=graph_label,
        )

    try:
        result = store.query_concept_region(limit_nodes=300, limit_edges=600)
    except Exception as exc:
        logger.warning("concept_atlas_network_query_failed error=%s", exc)
        return _unavailable(
            "substrate_store_error",
            str(exc),
            nodes=[],
            edges=[],
            god_node_count=0,
            component_count=0,
            graph=graph_label,
        )

    nodes = list(result.slice.nodes)
    edges = list(result.slice.edges)

    # --- filters: degrade sanely on malformed input, never 500 ---
    scope_norm = scope.strip() if isinstance(scope, str) else ""
    if scope_norm and scope_norm in _VALID_ANCHOR_SCOPES:
        nodes = [n for n in nodes if n.anchor_scope == scope_norm]
    elif scope_norm:
        logger.info("concept_atlas_network_ignored_bad_scope value=%s", scope_norm)

    min_act_value: Optional[float] = None
    if isinstance(min_activation, str) and min_activation.strip():
        try:
            parsed_min_act = float(min_activation)
        except ValueError:
            parsed_min_act = None
        # float("nan")/float("inf") parse without raising ValueError but are
        # not a meaningful activation threshold -- nan comparisons are always
        # False (silently empties the graph) and treating them as malformed
        # (i.e. ignored) is more honest than a confusing empty result.
        if parsed_min_act is None or not (0.0 <= parsed_min_act <= 1.0):
            logger.info("concept_atlas_network_ignored_bad_min_activation value=%s", min_activation)
        else:
            min_act_value = parsed_min_act
    if min_act_value is not None:
        nodes = [n for n in nodes if float(n.signals.activation.activation) >= min_act_value]

    node_ids = {n.node_id for n in nodes}

    # --- hydrate off-slice non-concept nodes (entities, etc.) reachable via
    # an edge whose other endpoint is a concept node already in this slice ---
    # store.query_concept_region() only ever returns concept-kind nodes in
    # `nodes` (see orion/substrate/store.py's read_concept_region()), so any
    # edge endpoint not in node_ids at this point is necessarily a
    # non-concept node -- most commonly an EntityNodeV1 mention. Without
    # this, the AND-filter two lines below silently drops every such edge:
    # a pre-existing gap (topic->entity `associated_with` mentions have
    # always been invisible here -- see
    # orion/substrate/adapters/topic_foundry.py's module docstring) that
    # also hid this design's new Orion/Juniper/Claude landmark edges (see
    # docs/superpowers/specs/2026-08-20-concept-graph-landmark-connection-design.md).
    # Guarded to only add non-concept nodes (`node_kind != "concept"`
    # below) so this never re-admits a concept node the scope/min_activation
    # filters above excluded on purpose -- a hydrated node is exempt from
    # those filters by construction (it was never eligible for them), not
    # smuggled past them. Bounded by _NETWORK_HYDRATION_MAX_EXTRA_NODES.
    hydrated_count = 0
    for e in edges:
        if hydrated_count >= _NETWORK_HYDRATION_MAX_EXTRA_NODES:
            break
        for candidate_id in (e.source.node_id, e.target.node_id):
            if candidate_id in node_ids or hydrated_count >= _NETWORK_HYDRATION_MAX_EXTRA_NODES:
                continue
            try:
                candidate_node = store.get_node_by_id(candidate_id)
            except Exception as exc:  # pragma: no cover - defensive, never let a backend hiccup abort the route
                logger.info("concept_atlas_network_hydration_lookup_failed node_id=%s error=%s", candidate_id, exc)
                candidate_node = None
            if candidate_node is None or getattr(candidate_node, "node_kind", None) == "concept":
                continue
            nodes.append(candidate_node)
            node_ids.add(candidate_id)
            hydrated_count += 1

    edges = [e for e in edges if e.source.node_id in node_ids and e.target.node_id in node_ids]

    focus_norm = focus.strip() if isinstance(focus, str) else ""
    if focus_norm:
        focus_lower = focus_norm.lower()
        focus_id = next(
            (
                n.node_id
                for n in nodes
                if n.node_id == focus_norm or str(getattr(n, "label", "") or "").lower() == focus_lower
            ),
            None,
        )
        if focus_id is not None:
            neighbor_ids = {focus_id}
            for e in edges:
                if e.source.node_id == focus_id:
                    neighbor_ids.add(e.target.node_id)
                if e.target.node_id == focus_id:
                    neighbor_ids.add(e.source.node_id)
            nodes = [n for n in nodes if n.node_id in neighbor_ids]
            node_ids = {n.node_id for n in nodes}
            edges = [e for e in edges if e.source.node_id in node_ids and e.target.node_id in node_ids]
        else:
            logger.info("concept_atlas_network_focus_not_found value=%s", focus_norm)

    # --- degree (activation-weighted) computed fresh at request time ---
    degree: dict[str, float] = {n.node_id: 0.0 for n in nodes}
    for e in edges:
        weight = 1.0 + float(e.salience or 0.0)
        if e.source.node_id in degree:
            degree[e.source.node_id] += weight
        if e.target.node_id in degree:
            degree[e.target.node_id] += weight

    # canonical nodes (orion/substrate/seed_concepts.yaml's hand-authored,
    # human_verified golden anchors -- Orion, Juniper, Claude, the
    # Orion-Juniper relationship) are ALWAYS god nodes, regardless of degree.
    # Confirmed live 2026-08-22: pure-degree ranking buried every one of
    # these behind whatever topic-foundry HDBSCAN cluster happened to
    # accumulate the most same-day co_occurs_with edges -- an artifact of
    # topic-foundry's day-bucket co-occurrence proxy rewarding vocabulary
    # ubiquity (e.g. "Orion" appearing in nearly every chat window), not a
    # real ranking of what's actually load-bearing to Orion's identity. A
    # canonical node earns its authority from human-verified seeding, not
    # from edge count, so it must never be outranked into invisibility by
    # organically-clustered noise. Remaining top-N slots (if any) still go
    # to the highest-degree non-canonical nodes, same as before, so real
    # organic hubs can still surface alongside the golden anchors. This is
    # independent of (and complementary to) the landmark-connection design
    # (2026-08-20) that gives these nodes real degree in the first place --
    # that fixed isolation, this fixes ranking still being pure-degree once
    # connected.
    canonical_ids = {n.node_id for n in nodes if getattr(n, "promotion_state", None) == "canonical"}
    remaining_slots = max(0, _GOD_NODE_TOP_N - len(canonical_ids))
    ranked = sorted(
        (nid for nid in degree if degree[nid] > 0.0 and nid not in canonical_ids),
        key=lambda nid: degree[nid],
        reverse=True,
    )
    god_ids = canonical_ids | set(ranked[:remaining_slots])
    component_of = _compute_connected_components(nodes, edges)
    display_labels = _display_labels(nodes, edges)

    node_payload = [
        {
            "id": n.node_id,
            "label": display_labels[n.node_id],
            "node_kind": n.node_kind,
            "anchor_scope": n.anchor_scope,
            "promotion_state": n.promotion_state,
            "activation": float(n.signals.activation.activation),
            "salience": float(n.signals.salience),
            "confidence": float(n.signals.confidence),
            "degree": degree.get(n.node_id, 0.0),
            "god_node": n.node_id in god_ids,
            "component_id": component_of.get(n.node_id, 0),
            # topic-foundry's HDBSCAN cluster id, when this node came from an
            # ingested topic-foundry run (orion/substrate/adapters/topic_foundry.py
            # writes it into metadata as "topic_id" -- there is no dedicated
            # schema field for it). None for concepts from any other producer.
            "topic_id": n.metadata.get("topic_id") if isinstance(n.metadata, dict) else None,
            # Honest-label plumbing: topic-foundry's adapter (see
            # orion/substrate/adapters/topic_foundry.py::_derive_label) falls
            # back to a bare "topic_<id>" placeholder when a run produced
            # neither a real topic label nor keywords -- a real, non-blank
            # string that nonetheless carries no human meaning. Surfacing
            # both fields lets the UI render those honestly (e.g. muted /
            # unlabeled) instead of indistinguishable from a real induced or
            # golden-seeded concept.
            # Derived from provenance.producer, NOT metadata["source"].
            # `source` is not in falkor_codec's closed metadata allowlist, so
            # it does not survive a rehydrate (forced at most every
            # snapshot_force_refresh_ceiling_sec) -- under the live
            # SUBSTRATE_STORE_BACKEND=falkor every node read back therefore had
            # source=None and every one of these evaluated to "concept".
            # provenance_producer IS a native column and does survive:
            # confirmed live 2026-08-29, 43 concepts carry
            # producer='topic_foundry_adapter'. Same fact, from the field that
            # is actually persisted, rather than growing the allowlist.
            "origin": (
                "topic_foundry"
                if getattr(n.provenance, "producer", None) == _TOPIC_FOUNDRY_PRODUCER
                or (isinstance(n.metadata, dict) and n.metadata.get("source") == "orion-topic-foundry")
                else "concept"
            ),
            # Same hydration trap as "origin" above: gating on
            # metadata["source"] made this permanently False under the live
            # falkor backend, so a genuinely unlabeled "topic_<id>" cluster
            # rendered as if it were a real concept name -- the exact
            # dishonest-label case this field was added to prevent.
            "synthetic_label": bool(
                (
                    getattr(n.provenance, "producer", None) == _TOPIC_FOUNDRY_PRODUCER
                    or (isinstance(n.metadata, dict) and n.metadata.get("source") == "orion-topic-foundry")
                )
                and str(n.label or "").startswith("topic_")
            ),
        }
        for n in nodes
    ]
    edge_payload = [
        {
            "id": e.edge_id,
            "source": e.source.node_id,
            "target": e.target.node_id,
            "predicate": e.predicate,
            "confidence": float(e.confidence),
            "salience": float(e.salience),
        }
        for e in edges
    ]

    # Non-default backends (e.g. GraphDBSubstrateStore under
    # SUBSTRATE_STORE_BACKEND=graphdb) can return a non-raising result that is
    # nonetheless backed by a stale fallback snapshot after an upstream query
    # failure (see query_concept_region()'s own degraded/error fields). The
    # default in-memory store never sets these, but surface them honestly
    # when a backend does -- "available: true" alone would otherwise hide a
    # real degraded-data condition (runtime truth beats config truth).
    degraded = bool(getattr(result, "degraded", False))
    return {
        "available": True,
        "nodes": node_payload,
        "edges": edge_payload,
        "god_node_count": len(god_ids),
        "component_count": len(set(component_of.values())),
        "truncated": bool(result.truncated),
        "degraded": degraded,
        "degraded_error": getattr(result, "error", None) if degraded else None,
        "graph": graph_label,
    }


def _ingest_topic_foundry_run(
    *,
    store: Any,
    model_name: str,
    log_prefix: str,
    landmark_concept_ids: Optional[dict[str, str]] = None,
    speaker_concept_ids: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Core ingestion logic shared by the Orion route/scheduler step and the
    AI Town route/scheduler step below -- parameterized (2026-08-20) over
    which store to write into and which topic-foundry model's latest run to
    pull, rather than duplicated. Every failure mode, the segment/mention
    fetch degrade-to-empty behavior, and the typed-relation classification
    pass are identical for both callers; only the destination graph and
    source model differ. See ``concept_atlas_ingest_topic_foundry`` below
    for the full original docstring this logic used to carry directly.

    ``landmark_concept_ids`` (added 2026-08-20, see
    ``docs/superpowers/specs/2026-08-20-concept-graph-landmark-connection-design.md``)
    is passed straight through to
    ``map_topic_foundry_run_to_substrate()`` -- see that function's own
    docstring. Deliberately left ``None`` by the AI Town caller: there are no
    golden seed concepts written into the AI Town store to connect to (see
    the design doc's non-goals).

    Deliberately a sync ``def`` (not ``async def``): the underlying HTTP calls
    use the blocking ``requests`` library (see ``topic_foundry_client.py``),
    and FastAPI runs sync route handlers in a worker thread pool automatically
    -- an ``async def`` wrapper here would block the event loop for the
    duration of every topic-foundry round trip instead.

    Fetches the latest completed run + its topics + per-topic keywords from
    topic-foundry over HTTP, converts them via
    ``orion.substrate.adapters.topic_foundry.map_topic_foundry_run_to_substrate``,
    and writes the resulting concept/evidence nodes and edges into ``store``.

    Reachable two ways: as this HTTP route (an operator-triggered manual
    call, mirroring ``/api/substrate/review-runtime/debug-run``'s shape), and
    -- as of Gap 5 -- as a plain function call from ``main.py``'s
    ``_run_substrate_topic_foundry_scheduler`` background task (gated by
    ``SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED``, default off), which calls
    this once per scheduler tick after triggering a training run. Both call
    paths are safe: this function is a plain zero-argument sync ``def`` with
    no FastAPI dependency injection, so direct invocation behaves identically
    to going through the route. Every failure mode below degrades to an
    honest, non-500-shaped response rather than fabricating a success result
    (the scheduler path logs the same dict's fields rather than checking an
    HTTP status).

    ``segment_topic_map`` (the co-occurrence input -- see
    ``map_topic_foundry_run_to_substrate``'s docstring) is built here from a
    real ``GET /segments`` fetch, grouped by UTC-day bucket of each
    segment's ``start_at``. ``SegmentRecord`` carries no direct session/
    conversation id -- day-bucketing is the best available real (not
    fabricated) proxy for "same conversation window" without inventing new
    schema. This is what feeds real candidate pairs to
    ``_classify_typed_concept_relations`` below (which only ever looks at
    ``co_occurs_with`` edges): before this, that classifier had zero real
    input and never fired in production despite being fully wired (PR
    #1132). A segments-fetch failure degrades to an empty
    ``segment_topic_map`` (no ``co_occurs_with`` edges produced for this
    call) rather than aborting the route -- concept/evidence node ingestion
    below is independent of it.

    ``segment_topic_id_map`` (added 2026-07-28, ``segment_id -> topic_id``)
    is built from the same ``GET /segments`` fetch above -- no extra network
    call. It resolves ``GET /kg/edges?predicate=mentions`` results (fetched
    separately below) to the topic concept each mentioned entity belongs to.
    This is topic-foundry's real, LLM-enriched entity-mention data, which
    used to go out on the now-retired ``orion:kg:edge:ingest.v1`` bus channel
    (zero live consumers -- see ``orion/substrate/adapters/topic_foundry.py``
    module docstring) and now feeds ``EntityNodeV1`` construction directly
    here instead. A mentions-fetch failure degrades to an empty list (no
    entity nodes/edges produced for this call) rather than aborting the
    route, same contract as the segments fetch above.
    """
    if store is None:
        return _unavailable("substrate_store_unavailable", concepts_written=0, entities_written=0, edges_written=0)

    base_url = str(getattr(settings, "TOPIC_FOUNDRY_BASE_URL", "") or "").strip()
    if not base_url:
        return _unavailable(
            "topic_foundry_base_url_not_configured",
            concepts_written=0,
            entities_written=0,
            edges_written=0,
        )

    try:
        # model_name-scoped -- see trigger_topic_foundry_enrichment()'s
        # identical comment; without this, ingestion could keep pulling
        # topics from the old, unfiltered model's runs indefinitely.
        fetched = fetch_run_topics_and_keywords(base_url, model_name=model_name)
    except TopicFoundryClientError as exc:
        logger.warning("%s_ingest_topic_foundry_fetch_failed error=%s", log_prefix, exc)
        return _unavailable(
            "topic_foundry_fetch_failed",
            str(exc),
            concepts_written=0,
            entities_written=0,
            edges_written=0,
        )
    except Exception as exc:  # pragma: no cover - defensive, never let a client bug 500 this route
        logger.warning("%s_ingest_topic_foundry_unexpected_fetch_error error=%s", log_prefix, exc)
        return _unavailable(
            "topic_foundry_unexpected_error",
            str(exc),
            concepts_written=0,
            entities_written=0,
            edges_written=0,
        )

    run_id = fetched["run_id"]
    topics = fetched["topics"]
    keywords_by_topic = fetched["keywords_by_topic"]

    segment_topic_map: dict[str, list[int]] = {}
    segment_topic_id_map: dict[str, int] = {}
    # segment_id -> recorded speakers, straight off provenance. Same
    # GET /segments payload the two maps above are built from, so this costs
    # no extra request.
    segment_speakers: dict[str, list[str]] = {}
    segments_fetched = 0
    # fetch_segments_for_run caps at the API's own limit with no pagination
    # loop, so a full page means the run may well have more. Deliberately
    # conservative: an exactly-full page that happens to be the whole run gets
    # marked partial, which errs toward under-claiming a share rather than
    # presenting a slice as ground truth.
    segments_truncated = False
    try:
        segments = fetch_segments_for_run(base_url, run_id)
        segments_fetched = len(segments)
        segments_truncated = segments_fetched >= _SEGMENTS_FETCH_LIMIT
        for seg in segments:
            topic_id = seg.get("topic_id")
            start_at = seg.get("start_at")
            segment_id = seg.get("segment_id")
            if topic_id is None or start_at is None:
                continue
            try:
                topic_id_int = int(topic_id)
            except (TypeError, ValueError):
                continue
            if topic_id_int == _OUTLIER_TOPIC_ID:
                continue
            if segment_id is not None:
                segment_topic_id_map[str(segment_id)] = topic_id_int
                provenance = seg.get("provenance")
                if isinstance(provenance, dict):
                    speakers = provenance.get("speakers")
                    if isinstance(speakers, list):
                        segment_speakers[str(segment_id)] = [
                            str(sp).strip().lower() for sp in speakers if str(sp).strip()
                        ]
            day_bucket = _day_bucket_from_timestamp(start_at)
            if day_bucket is None:
                continue
            segment_topic_map.setdefault(day_bucket, []).append(topic_id_int)
    except TopicFoundryClientError as exc:
        logger.warning(
            "%s_ingest_topic_foundry_segments_fetch_failed run_id=%s error=%s", log_prefix, run_id, exc
        )
        # Degrade to empty segment_topic_map/segment_topic_id_map --
        # co_occurs_with edges and mention-derived entity edges just won't
        # be produced for this ingestion call; concept/evidence node
        # ingestion below proceeds normally regardless.
    except Exception as exc:  # pragma: no cover - defensive, never let a client bug abort the route
        logger.warning(
            "%s_ingest_topic_foundry_segments_unexpected_error run_id=%s error=%s", log_prefix, run_id, exc
        )

    mention_edges: list[dict[str, Any]] = []
    mentions_fetched = 0
    try:
        mention_edges = fetch_mention_edges_for_run(base_url, run_id)
        mentions_fetched = len(mention_edges)
    except TopicFoundryClientError as exc:
        logger.warning(
            "%s_ingest_topic_foundry_mentions_fetch_failed run_id=%s error=%s", log_prefix, run_id, exc
        )
        # Degrade to empty mention_edges -- no entity nodes/edges produced
        # for this call; concept/evidence/co_occurs_with ingestion above is
        # independent of it.
    except Exception as exc:  # pragma: no cover - defensive, never let a client bug abort the route
        logger.warning(
            "%s_ingest_topic_foundry_mentions_unexpected_error run_id=%s error=%s", log_prefix, run_id, exc
        )

    try:
        from orion.substrate.adapters.topic_foundry import map_topic_foundry_run_to_substrate

        record = map_topic_foundry_run_to_substrate(
            run_id=run_id,
            topics=topics,
            keywords_by_topic=keywords_by_topic,
            segment_topic_map=segment_topic_map,
            mention_edges=mention_edges,
            segment_topic_id_map=segment_topic_id_map,
            landmark_concept_ids=landmark_concept_ids,
            segment_speakers=segment_speakers,
            speaker_concept_ids=speaker_concept_ids,
            segments_truncated=segments_truncated,
        )
    except Exception as exc:  # pragma: no cover - the adapter itself never raises, but don't trust across the boundary
        logger.warning("%s_ingest_topic_foundry_adapter_failed run_id=%s error=%s", log_prefix, run_id, exc)
        return _unavailable(
            "topic_foundry_adapter_failed",
            str(exc),
            run_id=run_id,
            concepts_written=0,
            entities_written=0,
            edges_written=0,
        )

    participation_edges = sum(
        1
        for e in record.edges
        if getattr(e.provenance, "source_kind", None) == "topic_foundry.participation"
    )
    if participation_edges == 0 and speaker_concept_ids:
        # Loud, not silent: the caller asked for participation edges and got
        # none. Almost always a run trained before provenance.speakers existed.
        logger.warning(
            "%s_ingest_topic_foundry_no_participation_edges run_id=%s "
            "segments_fetched=%s segments_with_speakers=%s -- Orion/Juniper "
            "will have no seed edges (the mention path is retired for them)",
            log_prefix,
            run_id,
            segments_fetched,
            sum(1 for v in segment_speakers.values() if v),
        )

    if not record.nodes:
        return _unavailable(
            "topic_foundry_no_usable_topics",
            run_id=run_id,
            topics_fetched=len(topics),
            concepts_written=0,
            entities_written=0,
            edges_written=0,
        )

    counting_store = _CountingSubstrateStore(store)
    try:
        from orion.substrate.materializer import SubstrateGraphMaterializer
        from orion.substrate.reconcile import SubstrateIdentityResolver

        # Exact-label identity works through the resolver's legacy key path;
        # explicitly wiring the store also activates embedding matches whenever
        # a caller supplies metadata['concept_embedding'].
        materializer = SubstrateGraphMaterializer(
            store=counting_store,
            identity_resolver=SubstrateIdentityResolver(store=counting_store),
        )
        materializer.apply_record(record)
    except Exception as exc:
        logger.warning("%s_ingest_topic_foundry_store_write_failed run_id=%s error=%s", log_prefix, run_id, exc)
        return _unavailable(
            "substrate_store_write_failed",
            str(exc),
            run_id=run_id,
            concepts_written=counting_store.concepts_written,
            evidence_nodes_written=counting_store.evidence_nodes_written,
            entities_written=counting_store.entities_written,
            edges_written=counting_store.edges_written,
            skipped_nodes=len(counting_store.skipped_nodes),
            skipped_edges=counting_store.skipped_edges,
        )

    # Post-ingestion typed-relation classification (Phase 4 wiring): scans the
    # store (not counting_store -- this must not inflate edges_written above,
    # which reports only the materializer's own co_occurs_with/supports
    # writes) for co_occurs_with edges worth an LLM classification call and
    # writes any resulting typed edge directly. Best-effort, capped, never
    # raises -- see _classify_typed_concept_relations for the honest-0 vs
    # error distinction. Adds real latency: up to
    # _RELATION_CLASSIFICATION_PAIR_CAP sequential bounded LLM RPCs (see
    # concept_relation_classifier.py's module docstring and the cap comment
    # above) -- acceptable here since this route is a manual, operator-
    # triggered trigger, not a hot path, and the route already runs as a sync
    # def in FastAPI's threadpool (see this function's own docstring), so the
    # added synchronous LLM round trips do not block the event loop.
    # Second, independent guard on the same failure the consecutive-failure
    # breaker bounds. The breaker decides how long a dead store is retried;
    # this decides how the result is REPORTED, and either alone leaves a hole:
    # a store failing most-but-not-all writes never trips the breaker, yet
    # reporting it `available: true` is the "success response hiding a
    # fallback" this route has been bitten by before.
    #
    # The line is break-even -- more nodes failed than landed -- not a tuned
    # ratio. It is the point past which the run did not produce a usable graph,
    # and it is what separates the two cases this patch has to tell apart:
    #
    #   one unwritable KIND   many concepts/evidence land, a few entities do
    #                         not -> written > skipped -> available, with the
    #                         skips surfaced. Reporting this unavailable would
    #                         throw away the edges the resilience exists to save.
    #   a broken STORE        few or no writes land -> skipped >= written ->
    #                         unavailable, exactly as before this patch.
    #
    # Reason stays `substrate_store_write_failed`: from a caller's side the
    # store write did fail, and the pre-existing contract for that case (real
    # partial counts, never a lie of all zeros) is preserved rather than
    # replaced with a new reason string nothing knows.
    written_nodes = (
        counting_store.concepts_written
        + counting_store.evidence_nodes_written
        + counting_store.entities_written
    )
    if counting_store.skipped_nodes and len(counting_store.skipped_nodes) > written_nodes:
        logger.error(
            "%s_ingest_topic_foundry_mostly_failed run_id=%s written=%d skipped=%d kinds=%s",
            log_prefix, run_id, written_nodes, len(counting_store.skipped_nodes),
            sorted({str(i.get("node_kind")) for i in counting_store.skipped_nodes}),
        )
        return _unavailable(
            "substrate_store_write_failed",
            str((counting_store.skipped_nodes[0] or {}).get("error") or ""),
            run_id=run_id,
            concepts_written=counting_store.concepts_written,
            evidence_nodes_written=counting_store.evidence_nodes_written,
            entities_written=counting_store.entities_written,
            edges_written=counting_store.edges_written,
            skipped_nodes=len(counting_store.skipped_nodes),
            skipped_edges=counting_store.skipped_edges,
        )

    if counting_store.skipped_nodes or counting_store.skipped_edges:
        logger.warning(
            "%s_ingest_topic_foundry_degraded run_id=%s skipped_nodes=%d skipped_edges=%d kinds=%s",
            log_prefix, run_id, len(counting_store.skipped_nodes), counting_store.skipped_edges,
            sorted({str(i.get("node_kind")) for i in counting_store.skipped_nodes}),
        )

    typed_edges_written = _classify_typed_concept_relations(store)

    # Same counter owns success and failure accounting. Counts are successful
    # upserts (including merges), not unique durable nodes after merge.
    return {
        "available": True,
        "run_id": run_id,
        "topics_fetched": len(topics),
        "concepts_written": counting_store.concepts_written,
        "evidence_nodes_written": counting_store.evidence_nodes_written,
        "entities_written": counting_store.entities_written,
        "edges_written": counting_store.edges_written,
        "segments_fetched": segments_fetched,
        "segment_topic_map_buckets": len(segment_topic_map),
        # Makes the silent-zero visible in the payload the scheduler already
        # logs. `provenance.speakers` only exists on runs trained after
        # 2026-08-28, so an older run yields segments_with_speakers=0 ->
        # participation_edges=0 -- and since the mention path was retired for
        # these speakers unconditionally, no seed edges from either route. That
        # must not look identical to a healthy ingest.
        "segments_with_speakers": sum(1 for v in segment_speakers.values() if v),
        "participation_edges": participation_edges,
        "mentions_fetched": mentions_fetched,
        "typed_edges_written": typed_edges_written,
        # Nodes the store refused, and edges dropped because an endpoint was
        # one of them. A run that skipped anything is NOT a clean success, and
        # reporting available=true with these silently at zero-visibility is
        # exactly the "success response hiding a fallback" this route has been
        # bitten by before. `skipped_node_kinds` names which kinds so an
        # operator can tell "the store cannot persist X" from a transient
        # write error, without dumping every node id into the payload.
        "skipped_nodes": len(counting_store.skipped_nodes),
        "skipped_edges": counting_store.skipped_edges,
        "skipped_node_kinds": sorted(
            {str(item.get("node_kind")) for item in counting_store.skipped_nodes}
        ),
    }


# Speakers who are resolved from recorded segment provenance rather than from
# entity mentions in the text. Measured on the real corpus 2026-08-28: each of
# these speaks in 254/254 chat_history_log rows (100%) while being NAMED in
# only 28%/26% -- so mention-based resolution had a ~28% ceiling on a fact that
# is already a foreign key, and 0% actual recall (topic_foundry_edges has never
# held a row). Anyone in this set is deliberately EXCLUDED from the mention
# path: CLAUDE.md 0A, "kill means kill, no fallback to the thing being killed".
# Derived, never hand-listed: a speaker is participation-resolvable precisely
# when some dataset column is attributed to them. Hardcoding the set lets it
# drift from _TOPIC_FOUNDRY_COLUMN_SPEAKERS -- and a name present here but
# absent there is dropped from the mention path AND never appears in any
# segment's speakers, so it silently gets zero edges from either route.
_PARTICIPATION_RESOLVED_SPEAKERS = frozenset(_TOPIC_FOUNDRY_COLUMN_SPEAKERS.values())


def _seed_concept_ids() -> dict[str, str]:
    """``label.lower() -> node_id`` for every golden seed concept.

    Reads the seed fixture directly (``load_seed_concept_nodes()`` is
    pure/read-only -- it does not write to any store); never writes anything.
    Degrades to an empty dict on any fixture read/parse problem, matching this
    module's existing degrade-not-raise convention.
    """
    try:
        from orion.substrate.seed import load_seed_concept_nodes

        nodes, _edges = load_seed_concept_nodes()
        return {str(node.label).strip().lower(): node.node_id for node in nodes if node.label}
    except Exception as exc:  # pragma: no cover - defensive, never let a fixture bug abort ingestion
        logger.warning("concept_atlas_seed_concept_ids_failed error=%s", exc)
        return {}


def _speaker_concept_ids() -> dict[str, str]:
    """Seed ids for speakers resolvable from provenance, for
    ``map_topic_foundry_run_to_substrate()``'s ``speaker_concept_ids``.
    """
    return {
        label: node_id
        for label, node_id in _seed_concept_ids().items()
        if label in _PARTICIPATION_RESOLVED_SPEAKERS
    }


def _landmark_concept_ids() -> dict[str, str]:
    """Seed ids still resolved through ENTITY MENTIONS, for
    ``map_topic_foundry_run_to_substrate()``'s ``landmark_concept_ids``.

    Claude only, as of 2026-08-28. In `chat_history_log` Claude is a *subject
    of conversation*, not a participant -- 20 rows mention the name, exactly 1
    row has Claude as the responder -- so a mention edge is the correct
    semantics there and the wrong one for Orion and Juniper, who now come from
    ``_speaker_concept_ids()`` instead. Orion-graph ingestion only; see
    ``_ingest_topic_foundry_run``'s docstring for why the AI Town caller does
    not use either.
    """
    return {
        label: node_id
        for label, node_id in _seed_concept_ids().items()
        if label not in _PARTICIPATION_RESOLVED_SPEAKERS
    }


@router.post("/api/substrate/concepts/ingest-topic-foundry")
def concept_atlas_ingest_topic_foundry() -> dict[str, Any]:
    """Operator-triggered, on-demand ingestion of topic-foundry's latest
    completed run into the Orion concept graph (``SUBSTRATE_SEMANTIC_STORE``).

    Deliberately a sync ``def`` (not ``async def``): the underlying HTTP calls
    use the blocking ``requests`` library (see ``topic_foundry_client.py``),
    and FastAPI runs sync route handlers in a worker thread pool automatically
    -- an ``async def`` wrapper here would block the event loop for the
    duration of every topic-foundry round trip instead.

    Reachable two ways: as this HTTP route (an operator-triggered manual
    call, mirroring ``/api/substrate/review-runtime/debug-run``'s shape), and
    -- as of Gap 5 -- as a plain function call from ``main.py``'s
    ``_run_substrate_topic_foundry_scheduler`` background task (gated by
    ``SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED``, default off), which calls
    this once per scheduler tick after triggering a training run. Both call
    paths are safe: this function is a plain zero-argument sync ``def`` with
    no FastAPI dependency injection, so direct invocation behaves identically
    to going through the route. See ``_ingest_topic_foundry_run`` above for
    the full ingestion logic (segment/mention fetch, adapter conversion,
    materializer write, typed-relation classification) this delegates to.
    """
    return _ingest_topic_foundry_run(
        store=_get_substrate_store(),
        model_name=_TOPIC_FOUNDRY_MODEL_NAME,
        log_prefix="concept_atlas",
        landmark_concept_ids=_landmark_concept_ids(),
        speaker_concept_ids=_speaker_concept_ids(),
    )


@router.post("/api/substrate/concepts/ingest-topic-foundry-aitown")
def concept_atlas_ingest_topic_foundry_aitown() -> dict[str, Any]:
    """Same as ``concept_atlas_ingest_topic_foundry`` above, for AI Town's
    own concept graph (``SUBSTRATE_SEMANTIC_STORE_AITOWN``) -- pulls from the
    AI Town topic-foundry model/dataset instead of the Orion one. See the
    module constants and ``_get_aitown_substrate_store`` above, and
    ``docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-
    atlas-readability-design.md`` ("AI Town's own concept graph").
    """
    return _ingest_topic_foundry_run(
        store=_get_aitown_substrate_store(),
        model_name=_TOPIC_FOUNDRY_AITOWN_MODEL_NAME,
        log_prefix="concept_atlas_aitown",
    )


@router.get("/concept-atlas")
async def concept_atlas_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template_path = TEMPLATES_DIR / "concept_atlas.html"
    if not template_path.is_file():
        raise HTTPException(status_code=404, detail="concept_atlas_template_missing")
    template = template_path.read_text(encoding="utf-8")
    rendered = template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version())
    return HTMLResponse(
        content=rendered,
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# --- whole-graph structural summary -----------------------------------------
#
# DELIBERATELY NOT A REPLACEMENT FOR THE /network PAYLOAD'S OWN component ids.
# Those are computed by `_compute_connected_components` over the ALREADY
# FILTERED node/edge lists, which is the correct answer to "what is connected
# in the view you are looking at" -- swapping in engine-wide `algo.WCC` there
# would report components whose members the caller cannot see, because
# scope/min_activation/focus removed them. This route answers the other
# question: what shape is the whole graph in, unfiltered. Both are real; they
# are not duplicates.
#
# WHY THIS EXISTS AT ALL. The atlas page could already say "136 nodes, 461
# edges, 12 components" and none of that told Juniper what was in there. The
# numbers that do, all verified live 2026-08-29 and none of them previously
# visible anywhere:
#   * 12 components is 1 blob of 116 + 1 island of 10 + 10 singletons, and
#     every singleton is retired telemetry (`node:substrate.transport` among
#     them -- a metric CLAUDE.md records as retired outright, still holding a
#     node here).
#   * 307 of the 461 edges are `co_occurs_with`, a same-day co-occurrence
#     proxy. Over 56 concepts that is 19.9% of every possible pair, which is
#     why `communities()` returns exactly one community. The hairball is an
#     edge-semantics problem, not a layout problem, and `saturation` is the
#     number that says so.
#   * pageRank and betweenness disagree: Orion and Juniper are pageRank #1/#2
#     and absent from the betweenness top 8. `bridges` surfaces that.

_SUMMARY_TOP_N = 8
# MUST equal the number of pageRank rows concept-atlas.js renders. They were 5
# and 6: a node at pageRank rank 6 could then appear in the influence column
# AND under "Bridges", whose tooltip says "not top pageRank" -- the card
# contradicting itself on screen. One constant, asserted against the JS.
_SUMMARY_PAGERANK_COMPARISON_N = _SUMMARY_TOP_N


_ANALYTICS_CACHE: dict[tuple[str, str], Any] = {}


def _build_graph_analytics(graph: Optional[str]) -> tuple[Any, str]:
    """Build a read-only GraphAnalytics for the requested graph, or (None, label).

    Resolves the same config the substrate store builders already use rather
    than minting new keys -- there is no new env surface in this route.

    ``FALKORDB_URI``/``FALKORDB_SUBSTRATE_GRAPH`` come off ``settings`` because
    both are real Settings fields (app/settings.py) and both are in Hub's
    compose ``environment:`` allowlist; reading them with ``os.getenv`` instead
    works in the container but returns "" under any in-process caller that
    loads config through pydantic-settings, which is how this route first came
    back ``available=false`` in testing.

    ``FALKORDB_AITOWN_SUBSTRATE_GRAPH`` is deliberately still an ``os.getenv``
    read with the SAME default string
    ``orion/substrate/falkor_store.py::build_aitown_falkor_substrate_store_from_env``
    uses, so the two resolve to the same graph. app/settings.py has a standing
    note explaining that no Settings field exists for it precisely because
    nothing read it; keeping this an env read adds a consumer without adding
    the config surface that note was guarding against.
    """
    import os

    from orion.graph.analytics import GraphAnalytics
    from orion.graph.falkor_client import RedisGraphQueryClient

    requested = (graph or "").strip().lower()
    if requested in ("aitown", "ai_town", "ai-town"):
        label = "aitown"
        graph_name = str(os.getenv("FALKORDB_AITOWN_SUBSTRATE_GRAPH", "orion_substrate_aitown")).strip()
    else:
        label = "substrate"
        graph_name = str(getattr(settings, "FALKORDB_SUBSTRATE_GRAPH", "") or "orion_substrate").strip()

    uri = str(getattr(settings, "FALKORDB_URI", "") or "").strip()
    if not uri or not graph_name:
        return None, label
    # Cached per (uri, graph): RedisGraphQueryClient builds a redis.Redis, and
    # each one carries its own ConnectionPool. Constructing a fresh client per
    # request leaks a pool per request -- nothing here closes them.
    cache_key = (uri, graph_name)
    cached = _ANALYTICS_CACHE.get(cache_key)
    if cached is None:
        cached = GraphAnalytics(RedisGraphQueryClient(uri=uri, graph_name=graph_name, read_only=True))
        _ANALYTICS_CACHE[cache_key] = cached
    return cached, label


def _ranked_payload(ranked: Any) -> list[dict[str, Any]]:
    return [
        {"node_id": r.node_id, "label": r.label, "score": round(float(r.score), 6)}
        for r in ranked
    ]


# NOT named "summary": /api/substrate/concepts/summary is already taken by the
# stat-tile endpoint above (concept_atlas_summary), and a second route on the
# same path does not raise -- FastAPI silently serves whichever registered
# first, so the new one would have been dead on arrival. Caught by a route test
# asserting the payload rather than by the app failing to start.
@router.get("/api/substrate/concepts/structure")
async def concept_atlas_structure(graph: Optional[str] = Query(None)) -> dict[str, Any]:
    analytics, graph_label = _build_graph_analytics(graph)
    if analytics is None:
        return _unavailable("falkordb_uri_unset", graph=graph_label)

    def _collect() -> dict[str, Any]:
        """All the blocking Redis work, in one place, off the event loop.

        Every call below is a synchronous round trip -- roughly six of them,
        ~30ms on the live graph, and betweenness is O(V*E) so it grows with the
        graph rather than with the request. Running that inline in an
        ``async def`` stalls the whole loop for every other Hub request. Same
        finding, same fix as api_routes.py:1406 (review, 2026-08-22).

        Each ranking is collected independently: the module docstring's whole
        point is that procedure behaviour varies per build, so one measure
        raising must not blank the node/edge/component census that already
        succeeded. A failed measure comes back as an empty list plus a name in
        `degraded_measures`, which the payload reports honestly.
        """
        summary = analytics.summary()
        concept_count = analytics.node_count("Concept")
        rankings: dict[str, list[dict[str, Any]]] = {}
        degraded_measures: list[str] = []
        for measure in ("pagerank", "betweenness", "harmonic"):
            try:
                rankings[measure] = _ranked_payload(analytics.rank(measure, top_n=_SUMMARY_TOP_N))
            except Exception as exc:  # noqa: BLE001 - one dead procedure must not blank the card
                logger.warning(
                    "concept_atlas_structure_measure_failed graph=%s measure=%s error=%s",
                    graph_label, measure, exc,
                )
                rankings[measure] = []
                degraded_measures.append(measure)

        dominant = summary.dominant_edge_type
        # Saturation is a statement about CONCEPT PAIRS, so both halves of the
        # ratio must be about concept pairs. Counting the dominant type's raw
        # edges against C(concepts, 2) is wrong whenever that type does not
        # join two concepts: `supports` runs evidence -> concept and joins
        # exactly 0 concept pairs while holding 80 edges (measured live
        # 2026-08-29), which the old formula would have rendered as "5.2% of
        # every possible pair". An evidence-heavy run could push it past 100%.
        # connected_pair_count asks the graph instead of assuming.
        pair_count = analytics.connected_pair_count(dominant, label="Concept") if dominant else 0
        saturation = analytics.pair_saturation(pair_count, concept_count) if dominant else None
        return {
            "summary": summary,
            "concept_count": concept_count,
            "rankings": rankings,
            "degraded_measures": degraded_measures,
            "dominant": dominant,
            "dominant_pair_count": pair_count,
            "saturation": saturation,
        }

    try:
        collected = await asyncio.to_thread(_collect)
    except Exception as exc:  # noqa: BLE001 - never 500 an operator page
        logger.warning("concept_atlas_structure_failed graph=%s error=%s", graph_label, exc)
        return _unavailable("graph_analytics_error", str(exc), graph=graph_label)

    summary = collected["summary"]
    concept_count = collected["concept_count"]
    rankings = collected["rankings"]
    dominant = collected["dominant"]
    saturation = collected["saturation"]

    # The finding this route exists to make visible: nodes that hold the graph
    # together but do not top the influence ranking. Compared against pageRank
    # rather than against the /network route's god-node set on purpose -- god
    # nodes are partly hand-seeded (canonical concepts are god nodes by
    # authority, not by structure), so comparing to them would conflate "we
    # decided this matters" with "the graph's shape says this matters".
    top_pagerank_ids = {r["node_id"] for r in rankings.get("pagerank", [])[:_SUMMARY_PAGERANK_COMPARISON_N]}
    bridges = [
        r for r in rankings.get("betweenness", [])
        if r["node_id"] not in top_pagerank_ids and r["score"] > 0.0
    ]

    return {
        "available": True,
        "graph": graph_label,
        "node_count": summary.node_count,
        "concept_count": concept_count,
        "edge_count": summary.edge_count,
        "edge_type_counts": summary.edge_type_counts,
        "dominant_edge_type": dominant,
        "dominant_edge_pair_count": collected["dominant_pair_count"],
        "dominant_edge_saturation": (round(saturation, 4) if saturation is not None else None),
        "degraded_measures": collected["degraded_measures"],
        "component_count": summary.component_count,
        "largest_component_size": summary.largest_component_size,
        "singleton_count": summary.singleton_count,
        "components": [
            {
                "component_id": c.component_id,
                "size": c.size,
                "is_singleton": c.is_singleton,
                "sample_labels": list(c.sample_labels),
            }
            for c in summary.components
        ],
        "rankings": rankings,
        "bridges": bridges,
    }
