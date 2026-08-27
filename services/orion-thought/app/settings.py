from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

# One source of truth for the context-seed char cap (review finding: was
# independently hardcoded here AND in store.py, two 240s to keep in sync by
# hand). store.py's top-level imports are stdlib-only -- importing it here
# costs nothing and creates no cycle (store.py never imports this module).
from .store import MAX_REVERIE_CONTEXT_CHARS as _MAX_REVERIE_CONTEXT_CHARS

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logger = logging.getLogger("orion-thought.settings")

# orion-mind runs 3 sequential LLM phases (semantic → appraisal → stance), each
# capped by MIND_LLM_TIMEOUT_SEC (default 60s on the orion-mind service). A wall
# budget below ~3× that ceiling guarantees synthesis is cut off mid-pipeline and
# the Mind degrades to contract_only (the empty-shell cognition failure mode).
# See fix/mind-enrichment-wall-budget.
MIND_LLM_TIMEOUT_SEC_ASSUMED: float = 60.0
MIND_ENRICHMENT_PHASE_COUNT: int = 3
MIND_ENRICHMENT_MIN_VIABLE_WALL_MS: int = int(
    MIND_LLM_TIMEOUT_SEC_ASSUMED * MIND_ENRICHMENT_PHASE_COUNT * 1000
)


class ThoughtSettings(BaseSettings):
    service_name: str = Field("orion-thought", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(7155, alias="THOUGHT_PORT")

    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    channel_thought_request: str = Field(
        "orion:thought:request",
        alias="CHANNEL_THOUGHT_REQUEST",
    )
    channel_thought_artifact: str = Field(
        "orion:thought:artifact",
        alias="CHANNEL_THOUGHT_ARTIFACT",
    )
    channel_thought_result_prefix: str = Field(
        "orion:thought:result:",
        alias="CHANNEL_THOUGHT_RESULT_PREFIX",
    )
    channel_cortex_exec_request: str = Field(
        "orion:cortex:exec:request",
        validation_alias=AliasChoices("CHANNEL_CORTEX_EXEC_REQUEST", "CORTEX_EXEC_REQUEST_CHANNEL"),
        alias="CHANNEL_CORTEX_EXEC_REQUEST",
    )
    channel_cortex_exec_result_prefix: str = Field(
        "orion:exec:result",
        validation_alias=AliasChoices("CHANNEL_CORTEX_EXEC_RESULT_PREFIX", "CORTEX_EXEC_RESULT_PREFIX"),
        alias="CHANNEL_CORTEX_EXEC_RESULT_PREFIX",
    )
    # 360, not 120 (2026-08-26). A shorter default is NOT the safe direction for
    # this key: an under-budgeted stance does not fail closed, it reports
    # `turn_deferred` -- which reads as "Orion judged the moment wrong" and is
    # indistinguishable from it at the Hub layer. Measured live: a stance that
    # completed correctly at 122s was thrown away at 120.006s. Must stay under
    # Hub's own TIMEOUT_SEC=400 outer wait. See services/orion-thought/.env_example.
    stance_react_timeout_sec: float = Field(360.0, alias="STANCE_REACT_TIMEOUT_SEC")

    # --- Reverie: spontaneous-thought mode (Phase A, default-off) ---
    reverie_enabled: bool = Field(False, alias="ORION_REVERIE_ENABLED")
    reverie_interval_sec: float = Field(90.0, alias="ORION_REVERIE_INTERVAL_SEC")
    reverie_min_salience: float = Field(0.0, alias="ORION_REVERIE_MIN_SALIENCE")
    channel_reverie_thought: str = Field(
        "orion:reverie:thought",
        alias="CHANNEL_REVERIE_THOUGHT",
    )

    # --- Reverie semantic lift (v1, default-off) ---
    reverie_semantic_lift_enabled: bool = Field(
        False, alias="ORION_REVERIE_SEMANTIC_LIFT_ENABLED"
    )
    reverie_referent_max_age_hours: float = Field(
        24.0, alias="ORION_REVERIE_REFERENT_MAX_AGE_HOURS"
    )
    channel_reverie_cortex_exec_request: str = Field(
        "orion:cortex:exec:request:background",
        alias="CHANNEL_REVERIE_CORTEX_EXEC_REQUEST",
    )

    # --- Reverie perception context (default-off, read-only) ---
    # Feeds the most recent orion-vision-council narrative(s) from `vision_events`
    # into the reverie prompt as ungrounded sensory context -- reverie is
    # otherwise 100% blind to the camera (build_reverie_context only ever
    # carried coalition/loop state). Deliberately does NOT widen evidence_refs
    # grounding to vision event ids (SpontaneousThoughtV1.grounding_ids() stays
    # coalition-only) -- that is a bigger, separate schema question. This patch
    # is step one of "event substrate first": get the percept into the context
    # a human can read before building any prediction/outcome scorer on top of
    # it (see docs/superpowers/specs/2026-08-12-perception-frontier-design.md,
    # Movement III -- the scorer is explicitly deferred until there is real
    # percept-grounded interpretation text to design a checker against).
    reverie_perception_enabled: bool = Field(False, alias="ORION_REVERIE_PERCEPTION_ENABLED")
    reverie_perception_max_age_sec: float = Field(
        180.0, alias="ORION_REVERIE_PERCEPTION_MAX_AGE_SEC"
    )
    reverie_perception_max_events: int = Field(
        3, alias="ORION_REVERIE_PERCEPTION_MAX_EVENTS"
    )

    # --- Reverie expectation scoring (Movement III, default-off) ---
    # Closes the loop between imagination and reality (docs/superpowers/specs/
    # 2026-08-12-perception-frontier-design.md, Movement III). reverie's
    # optional `expectation` field is a falsifiable claim about the room, set
    # only when the narrate call has recent_percepts (ORION_REVERIE_
    # PERCEPTION_ENABLED) and the LLM chose to state one. When this flag is
    # on: (a) a newly-set expectation gets an `expectation_checkable_by`
    # window opened on it, and (b) every tick spends at most one bounded
    # judge-LLM call resolving the single most-overdue still-pending
    # expectation against a fresh percept (confirmed/disconfirmed) -- or, if
    # no fresh-enough percept exists, writes "unscored" with no LLM call at
    # all. Never fabricates a confirmed/disconfirmed verdict on stale or
    # ambiguous evidence -- same honesty discipline as PerceptionContextV1's
    # staleness gate (P4). Flag off means this is a complete no-op: no pending
    # scan, no judge call, no checkable window ever opened.
    reverie_expectation_scoring_enabled: bool = Field(
        False, alias="ORION_REVERIE_EXPECTATION_SCORING_ENABLED"
    )
    # How long an expectation stays open before it becomes eligible for
    # scoring -- loosely paced off P4's 900s staleness gate (long enough for a
    # plausible next percept to land) without being the same number.
    reverie_expectation_check_window_sec: float = Field(
        1800.0, alias="ORION_REVERIE_EXPECTATION_CHECK_WINDOW_SEC"
    )

    # --- Reverie chain (Phase C, default-off) ---
    reverie_chain_enabled: bool = Field(False, alias="ORION_REVERIE_CHAIN_ENABLED")
    reverie_chain_max_steps: int = Field(4, alias="ORION_REVERIE_CHAIN_MAX_STEPS")
    reverie_refractory_sec: float = Field(900.0, alias="ORION_REVERIE_REFRACTORY_SEC")
    reverie_drift_temp: float = Field(0.7, alias="ORION_REVERIE_DRIFT_TEMP")
    channel_reverie_chain: str = Field(
        "orion:reverie:chain",
        alias="CHANNEL_REVERIE_CHAIN",
    )

    # --- Reverie grounding (Phase D, default-off, read-only) ---
    reverie_ground_consolidation: bool = Field(
        False, alias="ORION_REVERIE_GROUND_CONSOLIDATION"
    )

    # --- Compaction request (Phase E, default-off, queue only) ---
    reverie_compaction_request_enabled: bool = Field(
        False, alias="ORION_REVERIE_COMPACTION_REQUEST_ENABLED"
    )
    channel_dream_compaction_request: str = Field(
        "orion:dream:compaction-request",
        alias="CHANNEL_DREAM_COMPACTION_REQUEST",
    )

    # --- Resonance tripwire (Phase H, default-off, observation only) ---
    reverie_resonance_alert_enabled: bool = Field(
        False, alias="ORION_REVERIE_RESONANCE_ALERT_ENABLED"
    )
    channel_reverie_resonance_alert: str = Field(
        "orion:reverie:resonance-alert",
        alias="CHANNEL_REVERIE_RESONANCE_ALERT",
    )
    # How many recent chain rows to scan for a runaway theme.
    reverie_resonance_window: int = Field(200, alias="ORION_REVERIE_RESONANCE_WINDOW")

    # --- Resonance health monitor (Phase H+, reuses reverie_resonance_alert_enabled) ---
    # Edge-triggered orion-notify paging when a theme's resonance is actually
    # worsening (violation_count climbing across its last 2 persisted samples),
    # not merely "an alert exists" -- a stale historical burst can keep
    # re-reporting the same old numbers for days as it ages out of the
    # detector's lookback window, and that must not page anyone.
    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")

    # --- Reverie VISUAL chain (Patch 2 of docs/superpowers/specs/2026-08-20-
    # reverie-visual-chain-design.md, default-off). A second, parallel reverie
    # chain: generate an image via orion-diffusion-host, re-observe it via
    # orion-vision-host's existing caption_frame task, persist both, and
    # carry the caption forward as prior_description -- the enforced
    # continuity column design doc §2 built specifically to avoid repeating
    # the text chain's dead next_focus/drift fields. Independent worker loop
    # from reverie_chain above; sequential single-flight by construction (see
    # visual_chain.py module docstring), not a check-and-set marker.
    visual_chain_enabled: bool = Field(False, alias="ORION_VISUAL_CHAIN_ENABLED")
    # Slower than the text chain's ~90s cadence (design doc §4: "slower cadence
    # than the text chain"). Real cadence is max(this, actual run duration) --
    # intentional, not a bug (design doc §4).
    visual_chain_interval_sec: float = Field(600.0, alias="ORION_VISUAL_CHAIN_INTERVAL_SEC")
    # orion-diffusion-host on Circe (services/orion-diffusion-host README --
    # HOST_PORT=8014, same tailnet address the LLM gateway's route table uses
    # for circe-worker-agent-1, the llama.cpp worker this port used to serve).
    diffusion_host_base_url: str = Field(
        "http://100.112.254.99:8014", alias="ORION_DIFFUSION_HOST_BASE_URL"
    )
    visual_chain_diffusion_timeout_sec: float = Field(
        30.0, alias="ORION_VISUAL_CHAIN_DIFFUSION_TIMEOUT_SEC"
    )
    # Content-addressed image storage (orion.reverie.visual_storage, design
    # doc §6). Overridable so tests never touch the real mount.
    visual_chain_storage_dir: str = Field(
        "/mnt/storage-lukewarm/orion/reverie-visual",
        alias="ORION_VISUAL_CHAIN_STORAGE_DIR",
    )
    # orion-percept-store: the cross-host hop that lets a generated image
    # (produced on circe) reach orion-vision-host's caption_frame task
    # (athena) without assuming a shared filesystem -- same mechanism
    # orion-vision-council's foveal probe and orion-vision-frame-router
    # already use (see runner.py::_load_image_from_percept_store). Same
    # literal container hostname orion-vision-host's own
    # VISION_PERCEPT_STORE_URL already resolves (both are athena-network
    # services), not the tailnet address.
    visual_chain_percept_store_url: str = Field(
        "http://orion-athena-percept-store:8000/percepts",
        alias="ORION_VISUAL_CHAIN_PERCEPT_STORE_URL",
    )
    visual_chain_percept_store_token: str | None = Field(
        None, alias="ORION_VISUAL_CHAIN_PERCEPT_STORE_TOKEN"
    )
    visual_chain_percept_upload_timeout_sec: float = Field(
        10.0, alias="ORION_VISUAL_CHAIN_PERCEPT_UPLOAD_TIMEOUT_SEC"
    )
    # Originally the shared orion:exec:request:VisionHostService channel
    # (design doc §3: caption_frame + percept_sha256 already captions any
    # image, no vision-host code change needed, only a new producer). Moved
    # to circe's dedicated Qwen2-VL lane (services/orion-vision-host/
    # docker-compose.circe-qwen.yml) 2026-08-26 after live-confirming
    # athena's shared BLIP-base instance cannot produce a caption that
    # clears sanitize_caption's quality bar for a generated (not
    # camera-captured) image -- 3/3 real ticks came back uncaptioned.
    # reply_to is built per-call as f"{prefix}:{corr_id}", matching
    # orion:vision:reply:*'s documented wildcard (orion/bus/channels.yaml).
    channel_vision_host_request: str = Field(
        "orion:exec:request:VisionHostService:circe-vl", alias="CHANNEL_VISION_HOST_REQUEST"
    )
    channel_vision_reply_prefix: str = Field(
        "orion:vision:reply", alias="CHANNEL_VISION_REPLY_PREFIX"
    )
    visual_chain_caption_timeout_sec: float = Field(
        60.0, alias="ORION_VISUAL_CHAIN_CAPTION_TIMEOUT_SEC"
    )
    # Patch 3 context-seed tunables (store.py::load_latest_reverie_
    # interpretation) -- were bare module constants/unbounded until now,
    # unlike every other tunable in this block. reverie_context_char_limit's
    # default is store.py's own MAX_REVERIE_CONTEXT_CHARS (imported below,
    # not a second hardcoded 240) -- one source of truth, one-directional
    # import (settings -> store). Safe: store.py never imports this settings
    # module (its own module docstring: "never the heavy orion.substrate
    # package this thin service does not ship"), and its own top-level
    # imports are stdlib-only -- nothing heavy runs just by importing it here.
    #
    # reverie_context_max_age_sec closes a real staleness gap: without it, a
    # stalled/disabled text-reverie worker (chain.py) leaves the same old
    # thought answering forever, woven into the diffusion prompt and shown
    # in the visual cockpit's own context_text field as "Orion is currently
    # thinking" long after it stopped being current. This does NOT bound the
    # Hub Reverie tab's separate Text sub-view (reverie_routes.py::
    # text_recent), which has no staleness filter of its own and is not
    # touched by this change -- an operator can still see an old thought
    # presented as the latest one there.
    #
    # 900s (not the felt_state_reader.py convention of 2x the text-reverie
    # chain's own ~90s tick = 180s, and not proposal-runtime's
    # load_recent_reverie_thought's 300s default -- both real, checked, not
    # reused here on purpose): those two consumers read on their OWN fast
    # cadence, so a tight window matched to the *producer's* tick makes
    # sense for them. This context-seed is read once per VISUAL chain run
    # (ORION_VISUAL_CHAIN_INTERVAL_SEC, default 600s) -- a 180s window would
    # reject a perfectly fresh thought on almost every single visual-chain
    # tick, since 600s > 180s. 900s = 1.5x this consumer's own poll interval,
    # the same "multiple of the consumer's cadence" convention felt_state_
    # reader.py already uses, just computed against the right cadence.
    #
    # gt=0 on both (review finding): without it, a 0 or negative
    # ORION_REVERIE_CONTEXT_MAX_AGE_SEC makes the SQL freshness clause
    # permanently unsatisfiable (silently degrading to "no context-seed,
    # ever" -- indistinguishable from the genuine no-data case), and a
    # negative ORION_REVERIE_CONTEXT_CHAR_LIMIT turns Python's negative-index
    # slicing into "keep everything except the last N chars" -- the exact
    # opposite of a cap. Both fail loud at settings load instead of silently
    # doing the wrong thing at read time.
    reverie_context_char_limit: int = Field(
        _MAX_REVERIE_CONTEXT_CHARS, alias="ORION_REVERIE_CONTEXT_CHAR_LIMIT", gt=0
    )
    reverie_context_max_age_sec: float = Field(
        900.0, alias="ORION_REVERIE_CONTEXT_MAX_AGE_SEC", gt=0
    )

    # --- Attention salience trace publish gate ---
    # 2026-07-31: `orion.substrate.attention.salience`'s hand-picked
    # SEED_WEIGHTS formula (this flag's original "v1 vs v2" reason for
    # existing) was killed and replaced with GWT-coalition Borda
    # rank-aggregation -- there is only one salience formula now. This
    # flag's one remaining real purpose is gating whether `reverie.py`
    # publishes/persists an `AttentionSalienceTraceV1` row at all (see
    # `run_reverie_once`). Kept under its original name/key to avoid
    # env-parity churn for a purely cosmetic rename.
    attention_salience_v2_enabled: bool = Field(False, alias="ORION_ATTENTION_SALIENCE_V2_ENABLED")
    channel_attention_salience_trace: str = Field(
        "orion:attention:salience:trace",
        alias="CHANNEL_ATTENTION_SALIENCE_TRACE",
    )

    # --- Mind stance enrichment (unified turn; default-off) ---
    # Runs orion-mind before stance_react and injects an advisory self/attention
    # coloring. Silent no-op unless orion-mind has MIND_LLM_SYNTHESIS_ENABLED=true
    # (a separate service — not visible to this service's env-parity check).
    #
    # Wall/timeout budget: orion-mind runs THREE sequential LLM phases (semantic →
    # appraisal → stance), each capped by MIND_LLM_TIMEOUT_SEC (default 60s on the
    # orion-mind service). A wall below ~3× that ceiling cuts synthesis off
    # mid-pipeline and forces contract_only degradation, so the wall default must
    # stay >= MIND_ENRICHMENT_MIN_VIABLE_WALL_MS and the HTTP read timeout must
    # exceed the wall (so Mind's own fail-open result is returned, not aborted).
    mind_enrichment_enabled: bool = Field(False, alias="ORION_THOUGHT_MIND_ENRICHMENT_ENABLED")
    mind_base_url: str = Field("http://mind:6611", alias="ORION_MIND_BASE_URL")
    mind_timeout_sec: float = Field(210.0, alias="ORION_THOUGHT_MIND_TIMEOUT_SEC")
    mind_wall_ms: int = Field(180_000, alias="ORION_THOUGHT_MIND_WALL_MS")
    mind_router_profile: str = Field("default", alias="ORION_THOUGHT_MIND_ROUTER_PROFILE")
    mind_max_response_bytes: int = Field(2_000_000, alias="ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES")
    mind_artifact_publish_enabled: bool = Field(
        False, alias="ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED"
    )
    mind_coloring_max_items: int = Field(3, alias="ORION_THOUGHT_MIND_COLORING_MAX_ITEMS")
    channel_mind_artifact: str = Field("orion:mind:artifact", alias="CHANNEL_MIND_ARTIFACT")

    # --- Reasoning activity projection (always-on consumer; harmless when idle) ---
    # Consume per-call ReasoningCallV1 telemetry and materialize a rolling-window
    # ReasoningActivityV1 for φ. The buffer is capped (max_calls) so memory is
    # bounded regardless of producer rate.
    channel_reasoning_call: str = Field(
        "orion:cognition:reasoning_call", alias="CHANNEL_REASONING_CALL"
    )
    reasoning_activity_window_sec: float = Field(
        120.0, alias="REASONING_ACTIVITY_WINDOW_SEC"
    )
    reasoning_activity_max_calls: int = Field(
        2000, alias="REASONING_ACTIVITY_MAX_CALLS"
    )


def mind_enrichment_config_warnings(s: "ThoughtSettings") -> list[str]:
    """Deterministic boot-time coherence checks for the Mind enrichment budget.

    Only meaningful when enrichment is enabled. Returns human-readable warnings
    for budget settings that would silently force contract_only degradation.
    """
    warnings: list[str] = []
    if not s.mind_enrichment_enabled:
        return warnings
    if s.mind_wall_ms < MIND_ENRICHMENT_MIN_VIABLE_WALL_MS:
        warnings.append(
            f"wall_too_small ORION_THOUGHT_MIND_WALL_MS={s.mind_wall_ms} < "
            f"min_viable={MIND_ENRICHMENT_MIN_VIABLE_WALL_MS}: 3-phase LLM synthesis "
            "will be cut off mid-pipeline and Mind will degrade to contract_only"
        )
    if s.mind_timeout_sec * 1000.0 <= s.mind_wall_ms:
        warnings.append(
            f"http_timeout_not_above_wall ORION_THOUGHT_MIND_TIMEOUT_SEC={s.mind_timeout_sec}s "
            f"(<= ORION_THOUGHT_MIND_WALL_MS={s.mind_wall_ms}ms): the HTTP client may abort "
            "before Mind returns its own fail-open result, losing diagnostics/artifact"
        )
    return warnings


settings = ThoughtSettings()
logger.info(
    "Loaded orion-thought settings service=%s v=%s port=%s",
    settings.service_name,
    settings.service_version,
    settings.port,
)
for _mind_cfg_warning in mind_enrichment_config_warnings(settings):
    logger.warning("mind_enrichment_config %s", _mind_cfg_warning)
